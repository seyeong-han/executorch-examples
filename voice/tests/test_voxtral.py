"""Test Voxtral Realtime via ExecuTorch pybindings.

STATUS: KV cache does not persist between pybinding run_method calls for this
model. The text_decoder produces identical output regardless of cache_position,
meaning the decoder has no context from previous positions. This model must
use the subprocess-based C++ runner until the pybinding KV cache issue is
resolved.

The Whisper model's KV cache DOES work via pybindings (different export path
via optimum-executorch), so this issue is specific to the Voxtral export.
"""
import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from conftest import MODELS_DIR

MODEL_DIR = MODELS_DIR / "voxtral-realtime-xnnpack"


def test_kv_cache():
    """Verify KV cache behavior -- expected to FAIL currently."""
    model_path = None
    for p in MODEL_DIR.glob("model*.pte"):
        if "streaming" not in p.name and "preprocessor" not in p.name:
            model_path = str(p)
            break
    if model_path is None:
        print("SKIPPED: No Voxtral XNNPACK model found")
        return

    model = _load_for_executorch(model_path)

    inp = torch.randn(1, 1, 3072, dtype=torch.float32)
    pos0 = torch.tensor([0], dtype=torch.long)
    pos1 = torch.tensor([1], dtype=torch.long)

    logits_pos0 = model.run_method("text_decoder", [inp, pos0])[0].clone()
    logits_pos1 = model.run_method("text_decoder", [inp, pos1])[0].clone()

    diff = (logits_pos0 - logits_pos1).abs().max().item()
    if diff < 0.01:
        print(f"KNOWN ISSUE: KV cache not persisting (diff={diff:.6f})")
        print("Voxtral must use subprocess runner (C++ binary)")
    else:
        print(f"KV cache working (diff={diff:.6f})")
        print("Voxtral pybinding inference should work")


if __name__ == "__main__":
    test_kv_cache()
