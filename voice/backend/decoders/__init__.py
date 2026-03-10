"""Python-native decoders using ExecuTorch pybindings.

These replace the subprocess-based runners for models where the pybindings
work correctly (Silero VAD, Whisper, Parakeet, Sortformer).
Voxtral requires the subprocess runner due to a KV cache persistence issue.
"""
import logging

logger = logging.getLogger(__name__)

_ops_registered = False


def ensure_ops_registered():
    global _ops_registered
    if _ops_registered:
        return
    try:
        from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
    except Exception:
        pass
    try:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch  # noqa: F401
    except Exception:
        pass
    try:
        from executorch.kernels import quantized  # noqa: F401
    except Exception:
        pass
    try:
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    except Exception:
        pass
    _ops_registered = True
    logger.info("ExecuTorch operator libraries registered")
