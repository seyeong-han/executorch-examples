from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..runners.base import DiarizationResult, TranscriptionResult, VadResult
from ..runners.parakeet import ParakeetRunner
from ..runners.silero_vad import SileroVadRunner
from ..runners.sortformer import SortformerRunner
from ..runners.voxtral import VoxtralRunner
from ..runners.whisper import WhisperRunner

router = APIRouter(tags=["audio"])
logger = logging.getLogger(__name__)

RUNNER_CLASSES = {
    "whisper": WhisperRunner,
    "parakeet-tdt": ParakeetRunner,
    "voxtral-realtime": VoxtralRunner,
    "sortformer": SortformerRunner,
    "silero-vad": SileroVadRunner,
}


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
                f"Model {model_id} not ready on any backend. "
                f"Download it first via POST /v1/models/{model_id}/download",
            )

    if not entry.is_ready(backend):
        raise HTTPException(
            400,
            f"Model {model_id} not ready on {backend}. "
            f"Runner available: {backend in entry.available_runners}, "
            f"Model downloaded: {backend in entry.downloaded_backends}",
        )

    runner_cls = RUNNER_CLASSES.get(entry.runner_key)
    if not runner_cls:
        raise HTTPException(500, f"No runner class for {entry.runner_key}")

    runner = runner_cls(
        runner_path=str(entry.get_runner_path(backend)),
        model_dir=entry.get_model_dir(backend),
        backend=backend,
    )
    return runner, backend


async def _save_upload(file: UploadFile) -> str:
    """Save uploaded audio and convert to 16kHz mono WAV if needed."""
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    if suffix.lower() == ".wav":
        return tmp.name

    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found, passing raw file to runner")
        return tmp.name

    wav_path = tmp.name + ".wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp.name,
                "-ar", "16000", "-ac", "1",
                "-f", "wav", wav_path,
            ],
            capture_output=True, check=True, timeout=30,
        )
        Path(tmp.name).unlink(missing_ok=True)
        return wav_path
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
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
    runner, resolved_backend = _resolve_model(request, model, backend)
    audio_path = await _save_upload(file)

    try:
        result: TranscriptionResult = runner.run(
            audio_path,
            temperature=temperature,
            timestamps=timestamp_granularities,
        )
    except RuntimeError as e:
        raise HTTPException(500, f"Runner error: {e}")
    finally:
        Path(audio_path).unlink(missing_ok=True)

    return {
        "text": result.text,
        "model": model,
        "backend": resolved_backend,
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
    runner, resolved_backend = _resolve_model(request, model, backend)
    audio_path = await _save_upload(file)

    try:
        result: DiarizationResult = runner.run(audio_path, threshold=threshold)
    except RuntimeError as e:
        raise HTTPException(500, f"Runner error: {e}")
    finally:
        Path(audio_path).unlink(missing_ok=True)

    return {
        "segments": result.segments,
        "num_speakers": result.num_speakers,
        "model": model,
        "backend": resolved_backend,
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
    runner, resolved_backend = _resolve_model(request, model, backend)
    audio_path = await _save_upload(file)

    try:
        result: VadResult = runner.run(audio_path, threshold=threshold)
    except RuntimeError as e:
        raise HTTPException(500, f"Runner error: {e}")
    finally:
        Path(audio_path).unlink(missing_ok=True)

    return {
        "segments": result.segments,
        "speech_ratio": result.speech_ratio,
        "model": model,
        "backend": resolved_backend,
        "performance": asdict(result.performance),
    }
