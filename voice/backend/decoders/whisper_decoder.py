from __future__ import annotations

import json
import time
import torch
from pathlib import Path
from executorch.extension.pybindings.portable_lib import _load_for_executorch

from . import ensure_ops_registered
from ..runners.base import PerformanceStats, TranscriptionResult

SOT_TOKEN = 50258
EOT_TOKEN = 50257
EN_TOKEN = 50259
TRANSCRIBE_TOKEN = 50359
NO_TIMESTAMPS_TOKEN = 50363
MAX_NEW_TOKENS = 128


class SimpleWhisperTokenizer:
    def __init__(self, tokenizer_dir: Path):
        with open(tokenizer_dir / "tokenizer.json") as f:
            data = json.load(f)
        vocab = data.get("model", {}).get("vocab", {})
        self.id_to_token = {v: k for k, v in vocab.items()}
        for tok in data.get("added_tokens", []):
            self.id_to_token[tok["id"]] = tok["content"]

    def decode(self, token_ids: list[int]) -> str:
        pieces = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, "")
            if token.startswith("<|") and token.endswith("|>"):
                continue
            pieces.append(token.replace("\u0120", " "))
        return "".join(pieces).strip()


class WhisperDecoder:
    def __init__(self, model_dir: Path):
        ensure_ops_registered()
        self.model = _load_for_executorch(str(model_dir / "model.pte"))
        preprocessor_path = model_dir / "whisper_preprocessor.pte"
        self.preprocessor = (
            _load_for_executorch(str(preprocessor_path))
            if preprocessor_path.exists()
            else None
        )
        self.tokenizer = SimpleWhisperTokenizer(model_dir)

    def run(self, audio: torch.Tensor, temperature: float = 0.0) -> TranscriptionResult:
        t_start = time.perf_counter()

        pad_to = 480000
        if audio.shape[0] < pad_to:
            audio = torch.nn.functional.pad(audio, (0, pad_to - audio.shape[0]))
        elif audio.shape[0] > pad_to:
            audio = audio[:pad_to]

        if self.preprocessor:
            mel = self.preprocessor.forward([audio])[0]
        else:
            mel = audio.unsqueeze(0)

        encoder_out = self.model.run_method("encoder", [mel])[0]

        prompt_tokens = [SOT_TOKEN, EN_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN]
        cache_pos = torch.tensor([0], dtype=torch.long)
        logits = None

        for pt in prompt_tokens:
            decoder_ids = torch.tensor([[pt]], dtype=torch.long)
            logits = self.model.run_method("text_decoder", [decoder_ids, encoder_out, cache_pos])[0]
            cache_pos = cache_pos + 1

        t_first_token = time.perf_counter() - t_start

        generated_tokens = []
        for _ in range(MAX_NEW_TOKENS):
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            if next_token == EOT_TOKEN:
                break
            generated_tokens.append(next_token)
            decoder_ids = torch.tensor([[next_token]], dtype=torch.long)
            cache_pos = cache_pos + 1
            logits = self.model.run_method("text_decoder", [decoder_ids, encoder_out, cache_pos])[0]

        elapsed = time.perf_counter() - t_start
        gen_time = elapsed - t_first_token
        text = self.tokenizer.decode(generated_tokens)

        return TranscriptionResult(
            text=text,
            performance=PerformanceStats(
                inference_time_s=elapsed,
                tokens_generated=len(generated_tokens),
                tokens_per_second=len(generated_tokens) / gen_time if gen_time > 0 else 0,
            ),
        )
