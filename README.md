# transcribe-call

Transcribe a call recording with named speakers, at the highest accuracy.

```sh
transcribe-call call.mov --names "Ann Lee,bob.ray" --terms "Acme,Zephyr"
```

Writes `call.transcript.md` and `call.transcript.json` next to the recording (`--out DIR` to change).

## How it works

| Step | Tool |
|---|---|
| Clean the audio | ffmpeg |
| Transcribe with word timestamps | mlx-whisper `large-v3` |
| Separate the voices | pyannote `speaker-diarization-community-1` |
| Name each voice | Zoom's green active-speaker border, with the tile label read by tesseract |
| Settle turns under 4 seconds | pyannote voice embeddings |

Turns that stay doubtful are marked `(speaker uncertain)`. The header lists participants who never spoke.

## Options

| Option | Purpose |
|---|---|
| `--names` | Participant names exactly as Zoom shows them. Without it, speakers are `Speaker 1..N` and the video is not scanned. |
| `--terms` | Product and person names, to bias spelling. |
| `--language` | Default `en`. Use `auto` to detect. |
| `--speakers N` | Exact speaker count, if known. |
| `--no-denoise` | Skip the audio clean-up filter. |
| `--video-fps` | Video scan rate. Default 2. |
| `--out DIR` | Output directory. |

## Requirements

- macOS on Apple Silicon (mlx-whisper and the videotoolbox decoder)
- `ffmpeg`, `tesseract`
- Python 3 with `mlx-whisper`, `pyannote.audio`, `torch`, `opencv-python`, `numpy`, `scipy`
- A Hugging Face login (`hf auth login`). Accept the terms for the gated pyannote models on their model pages first.

## Limits

- Speaker naming works on Zoom gallery view. Without a green border, speakers fall back to `Speaker N`.
- A 21 minute 4K call takes about 17 minutes. The video scan is the slowest step.
- Whisper can still mis-hear names. Pass them with `--terms`.
