from __future__ import annotations

import re

from .base import BaseRunner, TranscriptionResult


class ParakeetRunner(BaseRunner):
    def run(self, audio_path: str, **kwargs) -> TranscriptionResult:
        model_pte = self.model_dir / "model.pte"
        tokenizer = self.model_dir / "tokenizer.model"
        timestamps = kwargs.get("timestamps", "segment")

        cmd = [
            self.runner_path,
            "--model_path", str(model_pte),
            "--tokenizer_path", str(tokenizer),
            "--audio_path", audio_path,
            "--timestamps", timestamps,
        ]

        data_path = self.model_dir / "aoti_cuda_blob.ptd"
        if data_path.exists():
            cmd.extend(["--data_path", str(data_path)])

        proc = self._run(cmd)
        return self._parse(proc.stdout + proc.stderr)

    def _parse(self, output: str) -> TranscriptionResult:
        text = ""
        segments = []

        m = re.search(r"Transcribed text:\s*(.+)", output)
        if m:
            text = m.group(1).strip()

        for m in re.finditer(
            r"([\d.]+)s\s*-\s*([\d.]+)s\s*:\s*(.+)", output
        ):
            segments.append({
                "start": float(m.group(1)),
                "end": float(m.group(2)),
                "text": m.group(3).strip(),
            })

        stats = self._parse_stats(output)
        return TranscriptionResult(
            text=text,
            segments=segments,
            performance=stats,
            raw_output=output,
        )
