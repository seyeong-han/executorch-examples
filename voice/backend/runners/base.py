from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..config import IS_MAC, LIBOMP_PATH

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    inference_time_s: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0


@dataclass
class TranscriptionResult:
    text: str = ""
    segments: list[dict] = field(default_factory=list)
    performance: PerformanceStats = field(default_factory=PerformanceStats)
    raw_output: str = ""


@dataclass
class DiarizationResult:
    segments: list[dict] = field(default_factory=list)
    num_speakers: int = 0
    performance: PerformanceStats = field(default_factory=PerformanceStats)
    raw_output: str = ""


@dataclass
class VadResult:
    segments: list[dict] = field(default_factory=list)
    speech_ratio: float = 0.0
    performance: PerformanceStats = field(default_factory=PerformanceStats)
    raw_output: str = ""


class BaseRunner(ABC):
    def __init__(self, runner_path: str, model_dir: Path, backend: str):
        self.runner_path = runner_path
        self.model_dir = model_dir
        self.backend = backend

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if IS_MAC and LIBOMP_PATH:
            env["DYLD_LIBRARY_PATH"] = f"/usr/lib:{LIBOMP_PATH}"
        return env

    def _run(self, cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
        logger.info(f"Running: {' '.join(cmd[:4])}...")
        start = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=self._build_env(),
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        logger.info(f"Runner finished in {elapsed:.2f}s, exit={result.returncode}")
        if result.returncode != 0:
            logger.error(f"Runner stderr: {result.stderr[-500:]}")
            raise RuntimeError(
                f"Runner failed (exit {result.returncode}): "
                f"{result.stderr[-300:]}"
            )
        return result

    def _parse_stats(self, output: str) -> PerformanceStats:
        stats = PerformanceStats()
        m = re.search(r"Total inference time:\s+([\d.]+)", output)
        if m:
            stats.inference_time_s = float(m.group(1))
        m = re.search(r"Generated (\d+) tokens", output)
        if m:
            stats.tokens_generated = int(m.group(1))
        m = re.search(r"Generated \d+ tokens:.*Rate:\s+([\d.]+)", output)
        if m:
            stats.tokens_per_second = float(m.group(1))
        return stats

    @abstractmethod
    def run(self, audio_path: str, **kwargs):
        ...
