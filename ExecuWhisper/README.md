# ExecuWhisper

`ExecuWhisper` is a native macOS app for on-device dictation with Parakeet TDT on ExecuTorch + Metal and optional LFM2.5-350M formatting on ExecuTorch + MLX. It keeps the app workflow local:

- record audio from the microphone
- stop recording
- keep `parakeet_helper` warm and send the captured PCM directly for transcription
- optionally rewrite the transcript with `lfm25_formatter_helper`
- save manual recording transcripts to local history or paste formatted dictation text

Unlike `VoxtralRealtime`, this app still does **not** do live token streaming, wake-word detection, or `silero_vad`. System dictation is available in a batch-compatible form: the default shortcut is `Ctrl+Space`, users can customize it in Settings, and the overlay pastes the final formatted text when recording stops.

## Features

- Record-then-transcribe flow with local microphone capture
- Auto-detected microphone selection for both manual recording and system dictation
- Batch-compatible system dictation with a customizable global shortcut and floating overlay
- First-launch model download from `younghan-meta/Parakeet-TDT-ExecuTorch-Metal`
- Single smart formatting prompt backed by `younghan-meta/LFM2.5-ExecuTorch-MLX`
- Searchable session history with rename, pinning, and recency grouping
- Text replacements for product names, acronyms, and domain terms
- Snippets for exact-match dictated templates
- Session export to `.txt`, `.json`, and `.srt`
- Lightweight DMG packaging by default, with optional bundled-model builds

## Requirements

- macOS 14.0+
- Apple Silicon
- Xcode 16+
- Conda
- `xcodegen`
- `libomp`

Install the host tools:

```bash
brew install xcodegen libomp
```

## Usage

### First launch

The default app build is intentionally small. On first launch, `ExecuWhisper` downloads:

- `model.pte`
- `tokenizer.model`
- `formatter/lfm2_5_350m_mlx_4w.pte`
- `formatter/tokenizer.json`
- `formatter/tokenizer_config.json`

into:

```text
~/Library/Application Support/ExecuWhisper/models
```

Session history is stored at:

```text
~/Library/Application Support/ExecuWhisper/sessions.json
```

Replacements are stored at:

```text
~/Library/Application Support/ExecuWhisper/replacements.json
```

### Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Cmd+Shift+R` | Start recording / stop and transcribe |
| `Cmd+Shift+C` | Copy the current transcript |
| `Ctrl+Space` | Toggle system dictation by default; change it in Settings |

## Build From Source

### 1. Activate the Metal environment

```bash
conda create -n et-metal python=3.12 -y
conda activate et-metal
```

### 2. Build the Parakeet helper

`parakeet_helper` is provided by [pytorch/executorch#18861](https://github.com/pytorch/executorch/pull/18861). Until that PR lands, check it out before building:

```bash
cd ~/executorch
gh pr checkout https://github.com/pytorch/executorch/pull/18861
make parakeet-metal
```

The helper is expected at:

```text
~/executorch/cmake-out/examples/models/parakeet/parakeet_helper
```

### 3. Build the LFM2.5 formatter helper

```bash
cd ~/executorch
make lfm_2_5_formatter-mlx
```

The helper is expected at:

```text
~/executorch/cmake-out/examples/models/llama/lfm25_formatter_helper
```

### 4. Build the macOS app

```bash
cd /Users/younghan/executorch-examples/ExecuWhisper
./scripts/build.sh
```

That produces:

```text
./build/Build/Products/Release/ExecuWhisper.app
```

### Optional: download or bundle models during the build

Download model artifacts into `MODEL_DIR` before building:

```bash
./scripts/build.sh --download-models
```

This downloads Parakeet from [younghan-meta/Parakeet-TDT-ExecuTorch-Metal](https://huggingface.co/younghan-meta/Parakeet-TDT-ExecuTorch-Metal) and the formatter artifacts from `younghan-meta/LFM2.5-ExecuTorch-MLX`.

Build a self-contained `.app` that already includes Parakeet and LFM2.5 formatter artifacts:

```bash
./scripts/build.sh --bundle-models
```

You can override the default paths with:

```bash
export EXECUTORCH_PATH="$HOME/executorch"
export PARAKEET_HELPER_PATH="/Users/younghan/project/executorch/cmake-out/examples/models/parakeet/parakeet_helper"
export FORMATTER_HELPER_PATH="$HOME/executorch/cmake-out/examples/models/llama/lfm25_formatter_helper"
export FORMATTER_METALLIB_PATH="$HOME/executorch/cmake-out/examples/models/llama/mlx.metallib"
export MODEL_DIR="$HOME/parakeet_metal"
export FORMATTER_MODEL_DIR="$HOME/lfm2_5_mlx"
```

## Create A DMG

After building the app:

```bash
./scripts/create_dmg.sh \
  "./build/Build/Products/Release/ExecuWhisper.app" \
  "./ExecuWhisper.dmg"
```

Behavior:

- If the app bundle contains only the helpers and runtime libraries, the DMG stays lightweight and the app downloads models on first launch.
- If the app bundle already contains Parakeet and LFM2.5 artifacts, the script validates all files and creates a bundled-model DMG.

## Run Tests

```bash
xcodegen generate
xcodebuild \
  -project ExecuWhisper.xcodeproj \
  -scheme ExecuWhisper \
  -derivedDataPath build \
  -destination "platform=macOS" \
  test
```

Current regression coverage includes:

- helper reuse and restart behavior in the warm bridge
- direct PCM handoff from the recorder into the helper
- preload and unload state handling for the helper lifecycle
- session compatibility for older `sessions.json` payloads
- replacement pipeline behavior
- LFM2.5 formatter prompt construction, protocol, bridge reuse, and fallback behavior
- session history grouping and pinning logic
- export rendering and file writing

## Manual Latency Gate

Use the helper benchmark to compare the first cold request against a second warm
request on the same helper process:

```bash
python3 ./scripts/benchmark_helper.py \
  --helper "$HOME/executorch/cmake-out/examples/models/parakeet/parakeet_helper" \
  --model "$HOME/parakeet_metal/model.pte" \
  --tokenizer "$HOME/parakeet_metal/tokenizer.model" \
  --audio /path/to/16khz_mono_float32.wav
```

Notes:

- The script exits non-zero if the warm request is not faster than the cold request.
- Pass `--min-speedup-ratio 0.2` to require at least a 20% warmup win.
- If you do not have a sample WAV handy, omit `--audio` and the script will use a generated synthetic clip.

## Troubleshooting

- `Parakeet helper not found`: check out [pytorch/executorch#18861](https://github.com/pytorch/executorch/pull/18861), then run `conda activate et-metal && cd ~/executorch && make parakeet-metal`
- `LFM2.5 formatter helper not found`: run `conda activate et-mlx && cd ~/executorch && make lfm_2_5_formatter-mlx`
- `mlx.metallib not found`: rerun `conda activate et-mlx && cd ~/executorch && make lfm_2_5_formatter-mlx`; the app bundles `mlx.metallib` next to `lfm25_formatter_helper`
- `libomp.dylib not found`: run `brew install libomp`
- Model download fails on first launch: check network access and verify the Parakeet and LFM2.5 Hugging Face repos are reachable from your machine
- Accessibility repeatedly asks during Xcode development: use the Settings access prompt and grant `ExecuWhisper Paste Helper` (`org.pytorch.executorch.ExecuWhisper.PasteHelper`); ExecuWhisper installs this stable helper app under Application Support so rebuilds can keep auto-paste working.
- DMG script says bundled-model files are missing: rebuild with `./scripts/build.sh --bundle-models`, or create a lightweight DMG instead
- Existing history is visible even if model assets are currently missing: use the `Home` page to repair downloads while keeping old transcripts accessible from the sidebar
- To reset macOS Accessibility and Microphone permissions for the app during testing:

```bash
tccutil reset Accessibility org.pytorch.executorch.ExecuWhisper
tccutil reset Microphone org.pytorch.executorch.ExecuWhisper
```
