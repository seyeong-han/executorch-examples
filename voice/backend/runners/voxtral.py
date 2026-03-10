from __future__ import annotations

import re

from .base import BaseRunner, TranscriptionResult


class VoxtralRunner(BaseRunner):
    def run(self, audio_path: str, **kwargs) -> TranscriptionResult:
        model_pte = self._find_model_pte()
        preprocessor = self._find_preprocessor()
        tokenizer = self.model_dir / "tekken.json"
        streaming = kwargs.get("streaming", False)

        cmd = [
            self.runner_path,
            "--model_path", str(model_pte),
            "--tokenizer_path", str(tokenizer),
            "--audio_path", audio_path,
            "--temperature", str(kwargs.get("temperature", 0.0)),
        ]
        if preprocessor:
            cmd.extend(["--preprocessor_path", str(preprocessor)])
        if streaming:
            cmd.append("--streaming")

        data_path = self.model_dir / "aoti_cuda_blob.ptd"
        if data_path.exists():
            cmd.extend(["--data_path", str(data_path)])

        proc = self._run(cmd)
        return self._parse(proc.stdout + proc.stderr)

    def _find_model_pte(self) -> Path:
        for p in self.model_dir.glob("model*.pte"):
            if "preprocessor" not in p.name and "streaming" not in p.name:
                return p
        return self.model_dir / "model.pte"

    def _find_preprocessor(self) -> Path | None:
        for p in self.model_dir.glob("preprocessor*.pte"):
            return p
        return None

    def _parse(self, output: str) -> TranscriptionResult:
        text = ""
        lines = output.split("\n")
        for line in lines:
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("I ")
                and not stripped.startswith("E ")
                and not stripped.startswith("W ")
                and not stripped.startswith("PyTorch")
                and not stripped.startswith("---")
                and "executorch:" not in stripped
                and "Metal" not in stripped
            ):
                if "</s>" in stripped:
                    stripped = stripped.replace("</s>", "").strip()
                if stripped:
                    text = stripped
                    break

        stats = self._parse_stats(output)
        return TranscriptionResult(
            text=text,
            performance=stats,
            raw_output=output,
        )
