#!/bin/sh
# Install or update transcribe-call. Run it again to update.
#
#   curl -fsSL https://github.com/zoharbabin/transcribe-call/releases/latest/download/install.sh | sh
#
# The release workflow fills in VERSION and SHA256 for each release.
# TRANSCRIBE_CALL_WHEEL=<path> installs that local wheel instead (for testing).
set -eu

VERSION="@VERSION@"
SHA256="@SHA256@"
REPO="zoharbabin/transcribe-call"
PYTHON="3.13"

die() {
  echo "error: $*" >&2
  exit 1
}

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) ;;
  *) die "transcribe-call needs macOS on Apple Silicon (mlx-whisper)." ;;
esac

if [ -n "${TRANSCRIBE_CALL_WHEEL:-}" ]; then
  wheel="$TRANSCRIBE_CALL_WHEEL"
else
  case "$VERSION" in
    @*) die "this copy of install.sh is not from a release. Use the one-liner in the README." ;;
  esac
  download_dir="$(mktemp -d)"
  trap 'rm -rf "$download_dir"' EXIT
  wheel="$download_dir/transcribe_call-$VERSION-py3-none-any.whl"
  echo "Downloading transcribe-call $VERSION"
  curl -fsSL -o "$wheel" "https://github.com/$REPO/releases/download/v$VERSION/transcribe_call-$VERSION-py3-none-any.whl"
  # uv ignores a #sha256= fragment on direct URLs, so check the file here
  echo "$SHA256  $wheel" | shasum -a 256 -c - >/dev/null 2>&1 || die "checksum mismatch, nothing was installed."
fi

for tool in ffmpeg tesseract uv; do
  command -v "$tool" >/dev/null 2>&1 && continue
  command -v brew >/dev/null 2>&1 || die "$tool is missing and Homebrew is not installed. See https://brew.sh"
  echo "Installing $tool with Homebrew"
  brew install "$tool"
done

echo "Installing transcribe-call"
uv tool install --force --python "$PYTHON" "$wheel"

bin_dir="$(uv tool dir --bin)"
case ":$PATH:" in
  *":$bin_dir:"*) ;;
  *) echo "Add $bin_dir to your PATH: run 'uv tool update-shell', then restart the shell." ;;
esac

cat <<'EOF'

Done. One-time setup for the speaker models:
  1. Accept the terms on the pyannote model pages (speaker-diarization-community-1, wespeaker-voxceleb-resnet34-LM).
  2. Run: uvx --from huggingface-hub hf auth login

Then: transcribe-call call.mov --names "Ann Lee,bob.ray"
EOF
