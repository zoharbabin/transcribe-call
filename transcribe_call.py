"""Transcribe a call recording with speaker names, at the highest accuracy.

    transcribe-call RECORDING --names "Ann Lee,bob.ray" [--terms "Acme,Zephyr"]

Steps: ffmpeg clean-up -> mlx-whisper large-v3 -> pyannote diarization ->
Zoom's green active-speaker border names each voice -> voice embeddings settle
short turns. Writes RECORDING.transcript.md and .json next to the recording.

--names: write them as Zoom shows them under each tile. Without --names the
speakers are "Speaker 1..N" and the video is not scanned.
Needs `hf auth login` once (the pyannote models are gated).
"""
import argparse
import collections
import difflib
import json
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np

START = time.time()
SCAN_WIDTH = 1600  # video frames are scaled to this width for the border scan; box coordinates use this space
WHISPER_MODEL = 'mlx-community/whisper-large-v3-mlx'  # full large-v3, not turbo: more accurate on accents
# community-1 needs no access to the gated segmentation-3.0 repo that speaker-diarization-3.1 pulls in
DIARIZATION_MODEL = 'pyannote/speaker-diarization-community-1'
EMBEDDING_MODEL = 'pyannote/wespeaker-voxceleb-resnet34-LM'
# rumble cut, steady-noise reduction, lift quiet speech. Heavy denoising can hurt Whisper (--no-denoise)
DENOISE = 'highpass=f=80,afftdn=nf=-25,dynaudnorm=f=150:g=15'
SHORT_TURN = 4.0  # seconds; shorter turns are re-checked by voice embedding
UNSURE_SIM = 0.35  # cosine similarity below this marks a short turn as uncertain
# MLX keeps freed GPU buffers for reuse. Uncapped, that cache grew with the audio length (17 GB for 30 minutes,
# about 32 GB for a 72 minute call) and the Mac ran out of memory. At 1 GiB, memory stays flat at the same speed.
MLX_CACHE_LIMIT = 1 << 30
# Voice embedding memory grows with clip length and turns can last minutes (17 GB on a 72 minute call).
# The first 30 seconds of a turn is enough to identify the voice.
MAX_EMBED_SECONDS = 30


def log(msg):
    """Progress line on stderr, prefixed with seconds since start."""
    print(f'[{time.time() - START:6.0f}s] {msg}', file=sys.stderr, flush=True)


def run(cmd, **kw):
    """Run a command, raise on failure, capture output."""
    return subprocess.run(cmd, check=True, capture_output=True, **kw)


def clock(t):
    """Seconds to mm:ss, or h:mm:ss from one hour."""
    h, rest = divmod(int(t), 3600)
    return f'{h}:{rest // 60:02d}:{rest % 60:02d}' if h else f'{rest // 60:02d}:{rest % 60:02d}'


def extract_audio(src, wav, denoise):
    """Write 16 kHz mono WAV (what Whisper and pyannote expect), optionally cleaned up."""
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src), '-vn', '-ac', '1', '-ar', '16000']
    run(cmd + (['-af', DENOISE] if denoise else []) + [str(wav)])


def transcribe(wav, language, prompt):
    """Whisper words with timestamps. The prompt (names, terms) biases spelling.

    No conditioning on previous text, so one bad passage does not spread. Word timestamps
    are required by the hallucination filter, which drops text invented over silence.
    """
    import mlx.core
    import mlx_whisper
    mlx.core.set_cache_limit(MLX_CACHE_LIMIT)
    result = mlx_whisper.transcribe(
        str(wav), path_or_hf_repo=WHISPER_MODEL, language=language, initial_prompt=prompt,
        word_timestamps=True, condition_on_previous_text=False, hallucination_silence_threshold=2)
    return [w for s in result['segments'] for w in s.get('words', [])]


def diarize(wav, min_speakers, max_speakers):
    """pyannote voice segments: [{start, end, speaker: 'SPEAKER_00'}, ...] in seconds."""
    import torch
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained(DIARIZATION_MODEL)
    pipe.to(torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))
    out = pipe(str(wav), min_speakers=min_speakers, max_speakers=max_speakers)
    ann = getattr(out, 'speaker_diarization', out)
    return [{'start': t.start, 'end': t.end, 'speaker': s} for t, _, s in ann.itertracks(yield_label=True)]


# ---- video: who has the green border -------------------------------------------------

def active_tile(frame):
    """(x, y, w, h) of Zoom's green active-speaker outline, or None.

    Keeps only long straight green runs, so green video content (plants, walls) is ignored.
    """
    r, g, b = (frame[..., i].astype(np.int16) for i in range(3))
    mask = ((g - r > 35) & (g - b > 35) & (g > 110)).astype(np.uint8)
    lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 40), np.uint8)) \
        | cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((30, 1), np.uint8))
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] >= 60 and b[3] >= 40]
    return max(boxes, key=lambda b: b[2] * b[3]) if boxes else None


def scan_video(src, fps, keyframes_only=False):
    """One row per sampled frame: {t, box, sig}.

    box is the active tile (or None). sig is a colour fingerprint of the tile centre; the
    same person gets a similar sig, which is how tiles are matched when the layout reorders.
    keyframes_only decodes just the keyframes (every few seconds) and holds each until the
    next one. That is far faster on 4K60, but the border is tracked less finely.
    """
    probe = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                 '-of', 'csv=p=0', str(src)], text=True).stdout.strip().split(',')
    width = SCAN_WIDTH
    height = round(width * int(probe[1]) / int(probe[0])) // 2 * 2
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', *(['-skip_frame', 'nokey'] if keyframes_only else []),
           '-hwaccel', 'videotoolbox', '-i', str(src),
           '-vf', f'fps={fps},scale={width}:{height}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * height * 3 * 4)
    rows = []
    while len(buf := proc.stdout.read(width * height * 3)) == width * height * 3:
        frame = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
        box = active_tile(frame)
        sig = None
        if box:
            x, y, w, h = box
            inner = frame[y + h // 10:y + h * 85 // 100, x + w // 10:x + w * 9 // 10].reshape(-1, 3) // 64
            hist = np.bincount(inner[:, 0] * 16 + inner[:, 1] * 4 + inner[:, 2], minlength=64).astype(np.float32)
            sig = np.sqrt(hist / hist.sum())
        rows.append({'t': len(rows) / fps, 'box': box, 'sig': sig})
    return rows


def read_label(src, t, box, tmp):
    """OCR the name label at the bottom-left of the tile. Returns text (may be noisy)."""
    png = run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-ss', str(t), '-i', str(src), '-frames:v', '1',
               '-f', 'image2pipe', '-vcodec', 'png', '-']).stdout
    frame = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    s = frame.shape[1] / SCAN_WIDTH
    x, y, w, h = box
    strip = frame[round((y + h - 18) * s):round((y + h - 2) * s), round((x + 2) * s):round((x + min(w - 4, 260)) * s)]
    dark = (strip < 110).mean(0) > 0.35  # the label sits on a dark strip; cut the crop where it ends
    gaps, end = 0, 0
    for i, d in enumerate(dark):
        gaps = 0 if d else gaps + 1
        if gaps > 12 * s:
            break
        end = i
    strip = cv2.resize(strip[:, :end + 1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() < 127:
        bw = 255 - bw
    img = Path(tmp) / 'label.png'
    cv2.imwrite(str(img), cv2.copyMakeBorder(bw, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255))
    return run(['tesseract', str(img), 'stdout', '--psm', '7'], text=True).stdout.strip()


def match_name(text, names):
    """Closest participant name to noisy OCR text, or None if nothing is close enough."""
    best = max(names, key=lambda n: difflib.SequenceMatcher(None, text.lower(), n.lower()).ratio())
    return best if difflib.SequenceMatcher(None, text.lower(), best.lower()).ratio() >= 0.6 else None


def video_track(src, rows, names, tmp):
    """Per scan frame: the participant name with the green border, or None.

    Tiles are clustered by appearance, then each cluster is named by a vote over OCR reads
    of a few of its frames. A single OCR read is too noisy; the vote and the fuzzy match are not.
    """
    idx = [i for i, r in enumerate(rows) if r['sig'] is not None]
    if len(idx) < 0.05 * len(rows):
        log('no green active-speaker border found; skipping video')
        return [None] * len(rows)
    X = np.array([rows[i]['sig'] for i in idx], np.float32)
    k = min(len(idx), 3 * len(names))  # several clusters per person: poses and layouts differ
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    cv2.setRNGSeed(0)
    _, labels, _ = cv2.kmeans(X, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    cluster_name = {}
    for c in range(k):
        members = [idx[j] for j in np.where(labels == c)[0]]
        if len(members) < 8:  # too short to matter
            continue
        sample = [members[int(q)] for q in np.linspace(0, len(members) - 1, min(8, len(members)))]
        votes = collections.Counter(filter(None, (match_name(read_label(src, rows[i]['t'], rows[i]['box'], tmp), names)
                                                  for i in sample)))
        if votes and votes.most_common(1)[0][1] >= 2:
            cluster_name[c] = votes.most_common(1)[0][0]
    log(f'video: {len(cluster_name)}/{k} appearance clusters named ' + str(sorted(set(cluster_name.values()))))
    track = [None] * len(rows)
    for j, i in enumerate(idx):
        track[i] = cluster_name.get(int(labels[j]))
    return track


def voice_names(segs, track, fps):
    """Name each pyannote voice by the participant whose border it overlaps most.

    Needs 40% overlap. Two voices may get the same name (one person split in two). Voices
    that never pass the bar stay "Speaker N". The border lags the voice, so this works on
    totals per voice, not on exact timing.
    """
    names = {}
    for voice in sorted({s['speaker'] for s in segs}):
        seen = collections.Counter()
        total = 0
        for s in segs:
            if s['speaker'] == voice:
                frames = track[int(s['start'] * fps):int(s['end'] * fps) + 1]
                total += len(frames)
                seen.update(n for n in frames if n)
        if seen and seen.most_common(1)[0][1] / total >= 0.4:
            names[voice] = seen.most_common(1)[0][0]
            log(f'{voice} -> {names[voice]} ({seen.most_common(1)[0][1] / total:.0%} of {total / fps:.0f}s)')
    for n, voice in enumerate(sorted({s['speaker'] for s in segs}), 1):
        names.setdefault(voice, f'Speaker {n}')
    return names


# ---- merge ---------------------------------------------------------------------------

def build_turns(words, segs, names):
    """Give each word the voice it overlaps most, then group runs into speaker turns.

    A pause over 2.5 seconds starts a new turn even for the same speaker.
    """
    def voice_of(w):
        overlap = [(min(w['end'], s['end']) - max(w['start'], s['start']), s['speaker']) for s in segs]
        best = max(overlap)
        if best[0] > 0:
            return best[1]
        mid = (w['start'] + w['end']) / 2
        return min(segs, key=lambda s: min(abs(s['start'] - mid), abs(s['end'] - mid)))['speaker']

    labels = [voice_of(w) for w in words]
    for i in range(1, len(labels) - 1):  # one-word flips between identical neighbours
        if labels[i - 1] == labels[i + 1] != labels[i]:
            labels[i] = labels[i - 1]
    turns = []
    for w, voice in zip(words, labels):
        if turns and turns[-1]['speaker'] == names[voice] and w['start'] - turns[-1]['end'] < 2.5:
            turns[-1]['text'] += w['word']
            turns[-1]['end'] = w['end']
        else:
            turns.append({'speaker': names[voice], 'start': w['start'], 'end': w['end'], 'text': w['word']})
    return turns


def refine_turns(wav_path, turns, track, fps):
    """Re-assign short turns to the closest voice profile; mark the doubtful ones.

    Profiles come from turns of 5+ seconds that the video agrees with. Short turns are where
    both pyannote and the lagging border go wrong. A turn is "unsure" if even its best match
    is weak, or the video still disagrees.
    """
    import torch
    from pyannote.audio import Inference, Model
    from scipy.io import wavfile

    sr, wav = wavfile.read(str(wav_path))
    wav = wav.astype(np.float32) / 32768.0
    inference = Inference(Model.from_pretrained(EMBEDDING_MODEL), window='whole')
    inference.to(torch.device('mps' if torch.backends.mps.is_available() else 'cpu'))

    def embed(t):
        a = t['start']
        b = min(max(t['end'], a + 0.5), a + MAX_EMBED_SECONDS)
        chunk = torch.from_numpy(wav[int(max(0, a - 0.05) * sr):int((b + 0.05) * sr)])[None]
        e = inference({'waveform': chunk, 'sample_rate': sr})
        return e / np.linalg.norm(e)

    def video_top(t):
        seen = collections.Counter(n for n in track[int(t['start'] * fps):int(t['end'] * fps) + 1] if n)
        return seen.most_common(1)[0][0] if seen else None

    profiles = collections.defaultdict(list)
    for t in turns:
        if t['end'] - t['start'] >= 5 and video_top(t) in (None, t['speaker']):
            profiles[t['speaker']].append(embed(t))
    profiles = {k: np.mean(v, 0) / np.linalg.norm(np.mean(v, 0)) for k, v in profiles.items()}
    for t in turns:
        t['unsure'] = False
        if t['end'] - t['start'] < SHORT_TURN and profiles:
            sims = {k: float(embed(t) @ p) for k, p in profiles.items()}
            t['speaker'] = max(sims, key=sims.get)
            t['unsure'] = max(sims.values()) < UNSURE_SIM or video_top(t) not in (None, t['speaker'])
    merged = []
    for t in turns:
        if merged and merged[-1]['speaker'] == t['speaker'] and t['start'] - merged[-1]['end'] < 2.5:
            merged[-1]['text'] += t['text']
            merged[-1]['end'] = t['end']
            merged[-1]['unsure'] = merged[-1]['unsure'] and t['unsure']
        else:
            merged.append(t)
    return merged


def write_outputs(src, out_dir, turns, names):
    """Write NAME.transcript.md and .json; the header lists named participants who never spoke."""
    base = Path(out_dir) / src.stem
    md = Path(f'{base}.transcript.md')
    spoke = sorted({t['speaker'] for t in turns})
    silent = [n for n in names if n not in spoke]
    lines = [f'# Transcript: {src.name}', '', 'Speakers: ' + ', '.join(spoke)]
    if silent:
        lines[-1] += '. Did not speak: ' + ', '.join(silent)
    lines += ['', 'Turns marked `(speaker uncertain)` are short and overlap a speaker change.', '']
    for t in turns:
        lines += [f'**{t["speaker"]}** [{clock(t["start"])}]' + (' (speaker uncertain)' if t['unsure'] else ''),
                  t['text'].strip(), '']
    md.write_text('\n'.join(lines))
    Path(f'{base}.transcript.json').write_text(json.dumps(turns, indent=1))
    return md


def main():
    ap = argparse.ArgumentParser(description='Transcribe a call recording with speaker names.')
    ap.add_argument('--version', action='version', version=f'transcribe-call {metadata.version("transcribe-call")}')
    ap.add_argument('recording', type=Path)
    ap.add_argument('--names', default='', help='comma-separated participant names as Zoom shows them')
    ap.add_argument('--terms', default='', help='comma-separated product or person names to bias spelling')
    ap.add_argument('--language', default='en', help='language code, or "auto" to detect (default: en)')
    ap.add_argument('--speakers', type=int, help='exact number of speakers, if known')
    ap.add_argument('--no-denoise', action='store_true', help='skip the audio clean-up filter')
    ap.add_argument('--video-fps', type=float, default=2, help='frames per second scanned for the speaker border')
    ap.add_argument('--fast-scan', action='store_true',
                    help='scan only video keyframes: much faster on 4K, but coarser speaker tracking')
    ap.add_argument('--out', type=Path, help='output directory (default: next to the recording)')
    args = ap.parse_args()
    src = args.recording.expanduser().resolve()
    if not src.is_file():
        sys.exit(f'not a file: {src}')
    names = [n.strip() for n in args.names.split(',') if n.strip()]
    prompt = ' '.join(filter(None, [
        f'A call between {", ".join(names)}.' if names else '',
        f'Terms: {args.terms}.' if args.terms else '']))

    with tempfile.TemporaryDirectory(prefix='transcribe-call-') as tmp:
        wav = Path(tmp) / 'clean.wav'
        log('extracting audio')
        extract_audio(src, wav, not args.no_denoise)
        log('transcribing (whisper large-v3)')
        words = transcribe(wav, None if args.language == 'auto' else args.language, prompt or None)
        log(f'{len(words)} words; diarizing')
        segs = diarize(wav, args.speakers or 2, args.speakers or max(len(names), 4))
        voices = sorted({s['speaker'] for s in segs})
        log(f'{len(voices)} voices found')
        fps = args.video_fps
        track = [None] * int(max(w['end'] for w in words) * fps + 2)
        if names:
            log('scanning video for the active speaker')
            rows = scan_video(src, fps, args.fast_scan)
            log(f'{sum(r["box"] is not None for r in rows)}/{len(rows)} frames have an active speaker; reading name labels')
            track = video_track(src, rows, names, tmp)
        voice_to_name = voice_names(segs, track, fps)
        log('merging and refining short turns')
        turns = refine_turns(wav, build_turns(words, segs, voice_to_name), track, fps)
        out_dir = args.out or src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        md = write_outputs(src, out_dir, turns, names)
    log(f'done: {md}')


if __name__ == '__main__':
    main()
