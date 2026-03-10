from __future__ import annotations

import re

from .base import BaseRunner, TranscriptionResult


class WhisperRunner(BaseRunner):
    def run(self, audio_path: str, **kwargs) -> TranscriptionResult:
        model_pte = self.model_dir / "model.pte"
        preprocessor = self.model_dir / "whisper_preprocessor.pte"
        tokenizer_dir = str(self.model_dir)

        cmd = [
            self.runner_path,
            "--model_path", str(model_pte),
            "--tokenizer_path", tokenizer_dir + "/",
            "--audio_path", audio_path,
            "--temperature", str(kwargs.get("temperature", 0.0)),
        ]
        if preprocessor.exists():
            cmd.extend(["--processor_path", str(preprocessor)])

        data_path = self.model_dir / "aoti_cuda_blob.ptd"
        if data_path.exists():
            cmd.extend(["--data_path", str(data_path)])

        proc = self._run(cmd)
        return self._parse(proc.stdout + proc.stderr)

    def _parse(self, output: str) -> TranscriptionResult:
        text = ""
        lines = output.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("<|") and "<|endoftext|>" in stripped:
                text = re.sub(r"<\|[^|]+\|>", "", stripped).strip()
                break
            if stripped.startswith("<|"):
                text = re.sub(r"<\|[^|]+\|>", "", stripped).strip()

        if not text:
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
                ):
                    text = stripped
                    break

        stats = self._parse_stats(output)
        return TranscriptionResult(
            text=text,
            performance=stats,
            raw_output=output,
        )
