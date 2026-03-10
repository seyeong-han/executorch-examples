from __future__ import annotations

import logging
from pathlib import Path

from ..config import (
    AVAILABLE_BACKENDS,
    CMAKE_OUT,
    HF_MODEL_CATALOG,
    MODELS_DIR,
    RUNNER_BINARIES,
)

logger = logging.getLogger(__name__)


class ModelEntry:
    def __init__(
        self,
        model_id: str,
        catalog: dict,
        available_runners: dict[str, Path],
        downloaded_backends: dict[str, Path],
    ):
        self.model_id = model_id
        self.task = catalog["task"]
        self.runner_key = catalog["runner"]
        self.display_name = catalog["display_name"]
        self.params = catalog["params"]
        self.hf_repos = catalog.get("repos", {})
        self.available_runners = available_runners
        self.downloaded_backends = downloaded_backends

    def is_ready(self, backend: str) -> bool:
        return (
            backend in self.available_runners
            and backend in self.downloaded_backends
        )

    def get_model_dir(self, backend: str) -> Path | None:
        return self.downloaded_backends.get(backend)

    def get_runner_path(self, backend: str) -> Path | None:
        return self.available_runners.get(backend)

    def to_dict(self) -> dict:
        backends = []
        for b in AVAILABLE_BACKENDS:
            backends.append({
                "name": b,
                "runner_available": b in self.available_runners,
                "model_downloaded": b in self.downloaded_backends,
                "hf_repo": self.hf_repos.get(b),
            })
        return {
            "id": self.model_id,
            "display_name": self.display_name,
            "task": self.task,
            "params": self.params,
            "backends": backends,
        }


class ModelRegistry:
    def __init__(self):
        self.models: dict[str, ModelEntry] = {}

    def scan(self):
        self.models.clear()
        runners = self._scan_runners()
        for model_id, catalog in HF_MODEL_CATALOG.items():
            runner_key = catalog["runner"]
            available_runners: dict[str, Path] = {}
            for backend, runner_path in runners.items():
                if runner_key in runner_path:
                    available_runners[backend] = Path(runner_path[runner_key])

            downloaded = self._scan_downloaded(model_id)

            self.models[model_id] = ModelEntry(
                model_id=model_id,
                catalog=catalog,
                available_runners=available_runners,
                downloaded_backends=downloaded,
            )

        ready = sum(
            1 for m in self.models.values()
            for b in AVAILABLE_BACKENDS if m.is_ready(b)
        )
        logger.info(
            f"Registry: {len(self.models)} models, "
            f"{ready} ready model-backend combos"
        )

    def _scan_runners(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for backend in AVAILABLE_BACKENDS:
            result[backend] = {}
            for runner_key, rel_path in RUNNER_BINARIES.items():
                full_path = CMAKE_OUT / rel_path
                if full_path.exists() and full_path.is_file():
                    result[backend][runner_key] = str(full_path)
        return result

    def _scan_downloaded(self, model_id: str) -> dict[str, Path]:
        downloaded: dict[str, Path] = {}
        for backend in AVAILABLE_BACKENDS:
            model_dir = MODELS_DIR / f"{model_id}-{backend}"
            if model_dir.exists() and any(model_dir.glob("*.pte")):
                downloaded[backend] = model_dir
        return downloaded

    def get(self, model_id: str) -> ModelEntry | None:
        return self.models.get(model_id)

    def list_all(self) -> list[dict]:
        return [m.to_dict() for m in self.models.values()]

    def list_ready(self) -> list[dict]:
        ready = []
        for m in self.models.values():
            for b in AVAILABLE_BACKENDS:
                if m.is_ready(b):
                    ready.append({**m.to_dict(), "active_backend": b})
                    break
        return ready
