# transcribe-call

Transcribe a call recording with named speakers, at the highest accuracy.

```sh
transcribe-call call.mov --names "Ann Lee,bob.ray" --terms "Acme,Zephyr"
```

Writes `call.transcript.md` and `call.transcript.json` next to the recording (`--out DIR` to change).

## Install and update

macOS on Apple Silicon only (mlx-whisper and the videotoolbox decoder).

```sh
curl -fsSL https://github.com/zoharbabin/transcribe-call/releases/latest/download/install.sh | sh
```

Run the same command again to update. It installs `ffmpeg`, `tesseract` and `uv` with Homebrew if they are missing, then installs the latest release into its own Python 3.13 environment. The download is checked against a SHA-256 baked into the installer.

One-time setup: accept the terms for the gated pyannote models on their model pages, then run `uvx --from huggingface-hub hf auth login`.

| Task | Command |
|---|---|
| Check the version | `transcribe-call --version` |
| Install a specific version | `curl -fsSL https://github.com/zoharbabin/transcribe-call/releases/download/vX.Y.Z/install.sh \| sh` |
| Uninstall | `uv tool uninstall transcribe-call` |

## Options

| Option | Purpose |
|---|---|
| `--names` | Participant names exactly as Zoom shows them. Without it, speakers are `Speaker 1..N` and the video is not scanned. |
| `--terms` | Product and person names, to bias spelling. |
| `--language` | Default `en`. Use `auto` to detect. |
| `--speakers N` | Exact speaker count, if known. |
| `--no-denoise` | Skip the audio clean-up filter. |
| `--video-fps` | Video scan rate. Default 2. |
| `--fast-scan` | Scan only video keyframes. About 13x faster on 4K60, but the border is tracked every few seconds instead of every `--video-fps`. |
| `--out DIR` | Output directory. |
| `--version` | Print the version and exit. |

## How it works

| Step | Tool |
|---|---|
| Clean the audio | ffmpeg |
| Transcribe with word timestamps | mlx-whisper `large-v3` |
| Separate the voices | pyannote `speaker-diarization-community-1` |
| Name each voice | Zoom's green active-speaker border, with the tile label read by tesseract |
| Settle turns under 4 seconds | pyannote voice embeddings |

Turns that stay doubtful are marked `(speaker uncertain)`. The header lists participants who never spoke.

## Limits

- Speaker naming works on Zoom gallery view. Without a green border, speakers fall back to `Speaker N`. Teams recordings have no green border: leave out `--names` to skip the scan.
- The video scan is the slowest step: about 10 seconds per minute of 4K60 video. The audio steps take about 14 minutes for a 72 minute call.
- Peak memory is about 10 GB at any call length.
- Whisper can still mis-hear names. Pass them with `--terms`.

## Development

The code is one module, `transcribe_call.py`. The unit tests need only numpy and OpenCV:

```sh
uv run --no-project --with numpy --with opencv-python-headless --with pytest pytest
```

CI also runs `ruff check .` and `shellcheck install.sh`.

### Releasing

The version is the git tag. Releases are final versions only (`vX.Y.Z`).

```sh
git tag v0.2.0 && git push origin v0.2.0
```

The `Release` workflow then runs lint and tests, builds the wheel, installs it on a macOS runner with `install.sh`, and publishes a GitHub Release. The release holds the wheel, the sdist, a version-stamped `install.sh` and `SHA256SUMS`. Fix a bad release by deleting it and its tag (`gh release delete vX.Y.Z --cleanup-tag`), then tag again.

## License

[MIT](LICENSE). The models it downloads (Whisper, pyannote) have their own licenses and terms.
