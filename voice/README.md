# ExecuTorch Voice Studio

A multi-backend, multi-OS web app for on-device speech AI with [ExecuTorch](https://github.com/pytorch/executorch).

Supports 5 voice models across XNNPACK (CPU), Metal (Apple GPU), and CUDA (NVIDIA GPU) backends.

## Models

| Model | Task | XNNPACK | Metal | CUDA |
|-------|------|---------|-------|------|
| [Whisper Tiny](https://huggingface.co/openai/whisper-tiny) | Transcription | yes | yes | planned |
| [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | Transcription + Timestamps | yes | yes | planned |
| [Voxtral Realtime 4B](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Streaming Transcription | yes | yes | planned |
| [Sortformer](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) | Speaker Diarization | yes | no | planned |
| [Silero VAD](https://github.com/snakers4/silero-vad) | Voice Activity Detection | yes | no | no |

## Quick Start

### 1. Install ExecuTorch and build runners

```bash
cd ~/executorch
./install_executorch.sh

# Build the runners you need (pick your backend)
make whisper-cpu parakeet-cpu sortformer-cpu silero-vad-cpu    # XNNPACK
make whisper-metal parakeet-metal voxtral_realtime-metal        # Metal (macOS)
```

### 2. Start the backend

```bash
cd voice/
pip install -r requirements.txt
EXECUTORCH_ROOT=~/executorch python -m backend.main
```

### 3. Start the frontend

```bash
cd voice/app/
npm install
npm run dev
```

Open http://localhost:5173

### 4. Download models

Click the download arrows next to model backend badges in the UI,
or use the API:

```bash
curl -X POST http://localhost:8000/v1/models/whisper-tiny/download \
  -H "Content-Type: application/json" \
  -d '{"backend": "xnnpack"}'
```

## API (OpenAI-compatible)

### Transcription

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=parakeet-tdt
```

### Diarization

```bash
curl -X POST http://localhost:8000/v1/audio/diarizations \
  -F file=@audio.wav \
  -F model=sortformer
```

### Voice Activity Detection

```bash
curl -X POST http://localhost:8000/v1/audio/vad \
  -F file=@audio.wav \
  -F model=silero-vad
```

### Streaming (WebSocket)

```javascript
const ws = new WebSocket("ws://localhost:8000/v1/realtime");
ws.onopen = () => ws.send(JSON.stringify({ model: "voxtral-realtime" }));
ws.onmessage = (e) => console.log(JSON.parse(e.data));
// Send raw 16kHz float32 PCM audio chunks via ws.send(audioBuffer)
```

## Architecture

```
Frontend (React + Vite + liquid-glass)
    |
    v
Backend (FastAPI + OpenAI-compatible API)
    |
    v
C++ Runners (subprocess per model)
    |
    v
ExecuTorch Runtime (XNNPACK / Metal / CUDA)
```

Pre-exported model files are hosted on [HuggingFace Hub](https://huggingface.co/younghan-meta)
and downloaded on demand.
