from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..models.downloader import (
    delete_model,
    download_model_with_progress,
)

router = APIRouter(tags=["models"])
logger = logging.getLogger(__name__)


@router.get("/models")
async def list_models(request: Request):
    from ..config import MODELS_DIR
    registry = request.app.state.registry
    return {"data": registry.list_all(), "object": "list", "models_dir": str(MODELS_DIR)}


@router.get("/models/{model_id}")
async def get_model(model_id: str, request: Request):
    registry = request.app.state.registry
    entry = registry.get(model_id)
    if not entry:
        raise HTTPException(404, f"Model not found: {model_id}")
    return entry.to_dict()


class DownloadRequest(BaseModel):
    backend: str = "xnnpack"


@router.post("/models/{model_id}/download")
async def download_model_endpoint(
    model_id: str, body: DownloadRequest, request: Request
):
    registry = request.app.state.registry
    entry = registry.get(model_id)
    if not entry:
        raise HTTPException(404, f"Model not found: {model_id}")
    if body.backend not in (entry.hf_repos or {}):
        raise HTTPException(
            400,
            f"Backend '{body.backend}' not available for {model_id}. "
            f"Available: {list(entry.hf_repos.keys())}",
        )

    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def _download():
        for event in download_model_with_progress(model_id, body.backend):
            queue.put_nowait(event)
        queue.put_nowait(None)

    async def event_stream():
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _download)

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

        registry.scan()
        yield f"data: {json.dumps({'status': 'done', 'progress': 100})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


class DeleteRequest(BaseModel):
    backend: str = "xnnpack"


@router.delete("/models/{model_id}")
async def delete_model_endpoint(
    model_id: str, request: Request, backend: str = "xnnpack"
):
    registry = request.app.state.registry
    entry = registry.get(model_id)
    if not entry:
        raise HTTPException(404, f"Model not found: {model_id}")

    deleted = delete_model(model_id, backend)
    if not deleted:
        raise HTTPException(404, f"No downloaded files for {model_id}/{backend}")

    registry.scan()
    return {"status": "deleted", "model_id": model_id, "backend": backend}
