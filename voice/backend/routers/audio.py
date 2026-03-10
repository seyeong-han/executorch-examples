from __future__ import annotations

import logging
import shutil
import struct
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..runners.base import DiarizationResult, TranscriptionResult, VadResult
from ..runners.voxtral import VoxtralRunner

router = APIRouter(tags=["audio"])
logger = logging.getLogger(__name__)

_decoder_cache: dict[str, object] = {}


def _get_decoder(runner_key: str, model_dir: Path):
    cache_key = f"{runner_key}:{model_dir}"
    if cache_key in _decoder_cache:
        return _decoder_cache[cache_key]

    if runner_key == "silero-vad":
        from ..decoders.silero_vad_decoder import SileroVadDecoder
        dec = SileroVadDecoder(model_dir)
    elif runner_key == "whisper":
        from ..decoders.whisper_decoder import WhisperDecoder
        dec = WhisperDecoder(model_dir)
    elif runner_key == "parakeet-tdt":
        from ..decoders.parakeet_decoder import ParakeetDecoder
        dec = ParakeetDecoder(model_dir)
    elif runner_key == "sortformer":
        from ..decoders.sortformer_decoder import SortformerDecoder
        dec = SortformerDecoder(model_dir)
    else:
        raise ValueError(f"No decoder for {runner_key}")

    _decoder_cache[cache_key] = dec
    logger.info(f"Loaded pybinding decoder: {runner_key}")
    return dec


def _resolve_model(request: Request, model_id: str, backend: str | None):
    registry = request.app.state.registry
    entry = registry.get(model_id)
    if not entry:
        raise HTTPException(404, f"Model not found: {model_id}")

    if backend is None:
        from ..config import AVAILABLE_BACKENDS
        for b in AVAILABLE_BACKENDS:
            if entry.is_ready(b):
                backend = b
                break
        if backend is None:
            raise HTTPException(
                400,
                f"Model {model_id} not ready. "
                f"Download it first via POST /v1/models/{model_id}/download",
            )

    if not entry.is_ready(backend):
        raise HTTPException(
            400,
            f"Model {model_id} not ready on {backend}.",
        )

    model_dir = entry.get_model_dir(backend)

    if entry.runner_key == "voxtral-realtime":
        runner = VoxtralRunner(
            runner_path=str(entry.get_runner_path(backend)),
            model_dir=model_dir,
            backend=backend,
        )
        return runner, backend, "subprocess"

    decoder = _get_decoder(entry.runner_key, model_dir)
    return decoder, backend, "pybinding"


def _load_audio_tensor(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        riff = f.read(4)
        if riff != b"RIFF":
            raise ValueError("Not a WAV file")
        f.read(4)
        f.read(4)
        bits_per_sample = 16
        data_bytes = b""
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
            elif chunk_id == b"data":
                data_bytes = f.read(chunk_size)
            else:
                f.read(chunk_size)
    if bits_per_sample == 16:
        samples = struct.unpack(f"<{len(data_bytes)//2}h", data_bytes)
        return torch.tensor(samples, dtype=torch.float32) / 32768.0
    raise ValueError(f"Unsupported WAV: {bits_per_sample}bit")


async def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    if suffix.lower() == ".wav":
        return tmp.name

    if not shutil.which("ffmpeg"):
        return tmp.name

    wav_path = tmp.name + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp.name, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, check=True, timeout=30,
        )
        Path(tmp.name).unlink(missing_ok=True)
        return wav_path
    except Exception:
        Path(wav_path).unlink(missing_ok=True)
        return tmp.name


@router.post("/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("whisper-tiny"),
    backend: str | None = Form(None),
    temperature: float = Form(0.0),
    timestamp_granularities: str = Form("segment"),
):
    runner, resolved_backend, mode = _resolve_model(request, model, backend)
    audio_path = await _save_upload(file)

    try:
        if mode == "pybinding":
            audio = _load_audio_tensor(audio_path)
            result: TranscriptionResult = runner.run(audio, temperature=temperature, timestamps=timestamp_granularities)
        else:
            result = runner.run(audio_path, temperature=temperature, timestamps=timestamp_granularities)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")
    finally:
        Path(audio_path).unlink(missing_ok=True)

    return {
        "text": result.text,
        "model": model,
        "backend": resolved_backend,
        "mode": mode,
        "segments": result.segments,
        "performance": asdict(result.performance),
    }


@router.post("/audio/diarizations")
async def diarize(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("sortformer"),
    backend: str | None = Form(None),
    threshold: float = Form(0.5),
):
    runner, resolved_backend, mode = _resolve_model(request, model, backend)
    audio_path = await _save_upload(file)

    try:
        if mode == "pybinding":
            audio = _load_audio_tensor(audio_path)
            result: DiarizationResult = runner.run(audio, threshold=threshold)
        else:
            result = runner.run(audio_path, threshold=threshold)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")
    finally:
        Path(audio_path).unlink(missing_ok=True)

    return {
        "segments": result.segments,
        "num_speakers": result.num_speakers,
        "model": model,
        "backend": resolved_backend,
        "mode": mode,
        "performance": asdict(result.performance),
    }


@router.post("/audio/vad")
async def voice_activity_detection(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("silero-vad"),
    backend: str | None = Form(None),
    threshold: float = Form(0.5),
):
    runner, resolved_backend, mode = _resolve_model(request, model, backend)
    audio_path = await _save_upload(file)

    try:
        if mode == "pybinding":
            audio = _load_audio_tensor(audio_path)
            result: VadResult = runner.run(audio, threshold=threshold)
        else:
            result = runner.run(audio_path, threshold=threshold)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")
    finally:
        Path(audio_path).unlink(missing_ok=True)

    return {
        "segments": result.segments,
        "speech_ratio": result.speech_ratio,
        "model": model,
        "backend": resolved_backend,
        "mode": mode,
        "performance": asdict(result.performance),
    }
