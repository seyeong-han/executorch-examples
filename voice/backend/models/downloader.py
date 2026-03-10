from __future__ import annotations

import logging
import shutil
import threading
import time
from collections.abc import Generator
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from ..config import HF_MODEL_CATALOG, MODELS_DIR

logger = logging.getLogger(__name__)


def _list_repo_files(hf_repo: str) -> list[dict]:
    api = HfApi()
    allowed_ext = {".pte", ".ptd", ".json", ".model", ".wav"}
    files = []
    for entry in api.list_repo_tree(hf_repo, recursive=True):
        if hasattr(entry, "rfilename") and hasattr(entry, "size"):
            if any(entry.rfilename.endswith(ext) for ext in allowed_ext):
                files.append({"name": entry.rfilename, "size": entry.size or 0})
    return files


def _dir_size(path: Path) -> int:
    try:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except Exception:
        return 0


def download_model_with_progress(
    model_id: str, backend: str
) -> Generator[dict, None, None]:
    catalog = HF_MODEL_CATALOG.get(model_id)
    if not catalog:
        yield {"status": "error", "error": f"Unknown model: {model_id}"}
        return

    repos = catalog.get("repos", {})
    hf_repo = repos.get(backend)
    if not hf_repo:
        yield {"status": "error", "error": f"No HF repo for {model_id}/{backend}"}
        return

    dest = MODELS_DIR / f"{model_id}-{backend}"
    dest.mkdir(parents=True, exist_ok=True)

    try:
        files = _list_repo_files(hf_repo)
    except Exception as e:
        yield {"status": "error", "error": f"Failed to list repo: {e}"}
        return

    total_bytes = sum(f["size"] for f in files)
    has_size_info = total_bytes > 0

    error_holder: list[Exception | None] = [None]
    current_file: list[str] = [""]
    file_index: list[int] = [0]

    def _do_download():
        try:
            for i, file_info in enumerate(files):
                current_file[0] = file_info["name"]
                file_index[0] = i + 1
                hf_hub_download(
                    repo_id=hf_repo,
                    filename=file_info["name"],
                    local_dir=str(dest),
                )
        except Exception as e:
            error_holder[0] = e

    thread = threading.Thread(target=_do_download, daemon=True)
    thread.start()

    while thread.is_alive():
        current_size = _dir_size(dest)
        if has_size_info:
            progress = min(int((current_size / total_bytes) * 100), 99)
        else:
            progress = min(int((file_index[0] / max(len(files), 1)) * 100), 99)
        yield {
            "status": "downloading",
            "progress": progress,
            "file": current_file[0],
            "file_index": file_index[0],
            "file_count": len(files),
            "downloaded_bytes": current_size,
            "total_bytes": total_bytes if has_size_info else 0,
        }
        time.sleep(0.5)

    if error_holder[0]:
        yield {"status": "error", "error": str(error_holder[0])}
        return

    final_size = _dir_size(dest)
    yield {
        "status": "complete",
        "progress": 100,
        "file": "",
        "file_index": len(files),
        "file_count": len(files),
        "downloaded_bytes": final_size,
        "total_bytes": total_bytes if has_size_info else final_size,
    }


def download_model(model_id: str, backend: str) -> Path:
    for event in download_model_with_progress(model_id, backend):
        if event["status"] == "error":
            raise ValueError(event["error"])
    return MODELS_DIR / f"{model_id}-{backend}"


def delete_model(model_id: str, backend: str) -> bool:
    dest = MODELS_DIR / f"{model_id}-{backend}"
    if dest.exists():
        shutil.rmtree(dest)
        logger.info(f"Deleted model: {dest}")
        return True
    return False
