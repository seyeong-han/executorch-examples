"""WebSocket endpoint for streaming transcription via Voxtral Realtime.

Follows the pattern of OpenAI's Realtime API: client sends audio chunks
over WebSocket, server streams back transcription text.

The backend pipes audio to the voxtral_realtime_runner's --mic mode via
stdin and reads stdout for transcribed text.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import AVAILABLE_BACKENDS, IS_MAC, LIBOMP_PATH

router = APIRouter(tags=["realtime"])
logger = logging.getLogger(__name__)


@router.websocket("/realtime")
async def realtime_transcribe(ws: WebSocket):
    await ws.accept()

    try:
        init_msg = await ws.receive_json()
    except WebSocketDisconnect:
        return

    model_id = init_msg.get("model", "voxtral-realtime")
    backend = init_msg.get("backend")

    registry = ws.app.state.registry
    entry = registry.get(model_id)
    if not entry:
        await ws.send_json({"error": f"Model not found: {model_id}"})
        await ws.close()
        return

    if backend is None:
        for b in AVAILABLE_BACKENDS:
            if entry.is_ready(b):
                backend = b
                break

    if not backend or not entry.is_ready(backend):
        await ws.send_json({"error": f"Model {model_id} not ready"})
        await ws.close()
        return

    runner_path = str(entry.get_runner_path(backend))
    model_dir = entry.get_model_dir(backend)

    model_pte = None
    for p in model_dir.glob("*streaming*.pte"):
        if "preprocessor" not in p.name:
            model_pte = p
            break
    if not model_pte:
        for p in model_dir.glob("model*.pte"):
            if "preprocessor" not in p.name:
                model_pte = p
                break

    preprocessor = None
    for p in model_dir.glob("preprocessor*streaming*.pte"):
        preprocessor = p
        break
    if not preprocessor:
        for p in model_dir.glob("preprocessor*.pte"):
            preprocessor = p
            break

    tokenizer = model_dir / "tekken.json"

    cmd = [
        runner_path,
        "--model_path", str(model_pte),
        "--tokenizer_path", str(tokenizer),
        "--mic",
    ]
    if preprocessor:
        cmd.extend(["--preprocessor_path", str(preprocessor)])

    import os
    env = dict(os.environ)
    if IS_MAC and backend == "metal" and LIBOMP_PATH:
        env["DYLD_LIBRARY_PATH"] = f"/usr/lib:{LIBOMP_PATH}"

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    await ws.send_json({"type": "session.created", "model": model_id, "backend": backend})

    async def read_stdout():
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if text and not text.startswith("I ") and not text.startswith("E "):
                try:
                    await ws.send_json({"type": "transcript.text", "text": text})
                except WebSocketDisconnect:
                    break

    reader_task = asyncio.create_task(read_stdout())

    try:
        while True:
            data = await ws.receive_bytes()
            if process.stdin and not process.stdin.is_closing():
                process.stdin.write(data)
                await process.stdin.drain()
    except WebSocketDisconnect:
        pass
    finally:
        if process.stdin and not process.stdin.is_closing():
            process.stdin.close()
        reader_task.cancel()
        process.terminate()
        await ws.send_json({"type": "session.ended"})
