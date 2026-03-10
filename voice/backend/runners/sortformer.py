from __future__ import annotations

import re

from .base import BaseRunner, DiarizationResult, PerformanceStats


class SortformerRunner(BaseRunner):
    def run(self, audio_path: str, **kwargs) -> DiarizationResult:
        model_pte = self.model_dir / "sortformer.pte"
        threshold = kwargs.get("threshold", 0.5)

        cmd = [
            self.runner_path,
            "--model_path", str(model_pte),
            "--audio_path", audio_path,
            "--threshold", str(threshold),
        ]

        data_path = self.model_dir / "aoti_cuda_blob.ptd"
        if data_path.exists():
            cmd.extend(["--data_path", str(data_path)])

        proc = self._run(cmd)
        return self._parse(proc.stdout + proc.stderr)

    def _parse(self, output: str) -> DiarizationResult:
        segments = []
        speakers = set()

        for m in re.finditer(
            r"\s+([\d.]+)\s+([\d.]+)\s+(speaker_\d+)", output
        ):
            speaker = m.group(3)
            speakers.add(speaker)
            segments.append({
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "speaker": speaker,
            })

        speech_match = re.search(r"(\d+) segments", output)
        stats = PerformanceStats()
        m = re.search(r"Total inference time:\s+([\d.]+)", output)
        if m:
            stats.inference_time_s = float(m.group(1))

        return DiarizationResult(
            segments=segments,
            num_speakers=len(speakers),
            performance=stats,
            raw_output=output,
        )
