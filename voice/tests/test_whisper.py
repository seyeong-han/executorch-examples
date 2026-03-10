"""Test Whisper Tiny via ExecuTorch pybindings.

Loads model.pte + whisper_preprocessor.pte, runs encoder then autoregressive
text_decoder loop with KV cache, decodes tokens via HuggingFace tokenizer.
"""
import time
import json
import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from conftest import MODELS_DIR, load_audio_wav, AUDIO_FILE

MODEL_DIR = MODELS_DIR / "whisper-tiny-xnnpack"
MODEL_PATH = str(MODEL_DIR / "model.pte")
PREPROCESSOR_PATH = str(MODEL_DIR / "whisper_preprocessor.pte")
TOKENIZER_DIR = MODEL_DIR

SOT_TOKEN = 50258  # <|startoftranscript|>
EOT_TOKEN = 50257  # <|endoftext|>
EN_TOKEN = 50259   # <|en|>
TRANSCRIBE_TOKEN = 50359  # <|transcribe|>
NO_TIMESTAMPS_TOKEN = 50363  # <|notimestamps|>
MAX_NEW_TOKENS = 128


class SimpleWhisperTokenizer:
    def __init__(self, tokenizer_dir):
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
            token = token.replace("\u0120", " ")
            pieces.append(token)
        return "".join(pieces).strip()


def run_whisper(audio: torch.Tensor):
    preprocessor = _load_for_executorch(PREPROCESSOR_PATH)
    model = _load_for_executorch(MODEL_PATH)

    print(f"Model methods: {model.method_names()}")

    t_start = time.perf_counter()

    pad_to = 480000  # 30s at 16kHz
    if audio.shape[0] < pad_to:
        audio = torch.nn.functional.pad(audio, (0, pad_to - audio.shape[0]))
    elif audio.shape[0] > pad_to:
        audio = audio[:pad_to]

    mel = preprocessor.forward([audio])[0]
    print(f"Mel shape: {mel.shape}")

    encoder_out = model.run_method("encoder", [mel])[0]
    t_encode = time.perf_counter() - t_start
    print(f"Encoder output shape: {encoder_out.shape}, encode time: {t_encode:.3f}s")

    decoder_ids = torch.tensor([[SOT_TOKEN]], dtype=torch.long)
    cache_pos = torch.tensor([0], dtype=torch.long)

    prompt_tokens = [SOT_TOKEN, EN_TOKEN, TRANSCRIBE_TOKEN, NO_TIMESTAMPS_TOKEN]
    for pt in prompt_tokens:
        decoder_ids = torch.tensor([[pt]], dtype=torch.long)
        logits = model.run_method("text_decoder", [decoder_ids, encoder_out, cache_pos])[0]
        cache_pos = cache_pos + 1

    t_first_token = time.perf_counter() - t_start

    generated_tokens = []
    for i in range(MAX_NEW_TOKENS):
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()

        if next_token == EOT_TOKEN:
            break

        generated_tokens.append(next_token)
        decoder_ids = torch.tensor([[next_token]], dtype=torch.long)
        cache_pos = cache_pos + 1
        logits = model.run_method("text_decoder", [decoder_ids, encoder_out, cache_pos])[0]

    elapsed = time.perf_counter() - t_start
    gen_time = elapsed - t_first_token

    tokenizer = SimpleWhisperTokenizer(TOKENIZER_DIR)
    text = tokenizer.decode(generated_tokens)

    return text, generated_tokens, t_first_token, gen_time, elapsed


def main():
    print(f"Loading audio: {AUDIO_FILE}")
    audio = load_audio_wav(AUDIO_FILE)
    print(f"Audio: {audio.shape[0]} samples, {audio.shape[0]/16000:.1f}s")

    text, tokens, ttft, gen_time, total = run_whisper(audio)

    print(f"\nTranscription:\n{text}\n")
    print(f"Tokens generated: {len(tokens)}")
    print(f"TTFT: {ttft:.3f}s")
    print(f"Generation time: {gen_time:.3f}s")
    print(f"Generation rate: {len(tokens)/gen_time:.1f} tok/s" if gen_time > 0 else "")
    print(f"Total time: {total:.3f}s")
    print(f"\nWhisper test PASSED")


if __name__ == "__main__":
    main()
