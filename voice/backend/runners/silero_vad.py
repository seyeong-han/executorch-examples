from __future__ import annotations

import re

from .base import BaseRunner, PerformanceStats, VadResult


class SileroVadRunner(BaseRunner):
    def run(self, audio_path: str, **kwargs) -> VadResult:
        model_pte = self.model_dir / "silero_vad.pte"
        threshold = kwargs.get("threshold", 0.5)

        cmd = [
            self.runner_path,
            "--model_path", str(model_pte),
            "--audio_path", audio_path,
            "--threshold", str(threshold),
        ]

        proc = self._run(cmd)
        return self._parse(proc.stdout + proc.stderr)

    def _parse(self, output: str) -> VadResult:
        segments = []

        for m in re.finditer(
            r"\s+([\d.]+)\s+([\d.]+)\s+speech", output
        ):
            segments.append({
                "start": float(m.group(1)),
                "end": float(m.group(2)),
            })

        speech_ratio = 0.0
        m = re.search(r"Speech:\s+\d+/\d+ frames \(([\d.]+)%\)", output)
        if m:
            speech_ratio = float(m.group(1)) / 100.0

        stats = PerformanceStats()
        m = re.search(r"Total inference time:\s+([\d.]+)", output)
        if m:
            stats.inference_time_s = float(m.group(1))

        return VadResult(
            segments=segments,
            speech_ratio=speech_ratio,
            performance=stats,
            raw_output=output,
        )
