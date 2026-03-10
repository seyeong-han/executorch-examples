"""Shared setup for all voice model pybinding tests."""
import torch

# Register operator libraries before any model loading.
# Order matters: quantized decomposed -> portable_lib -> quantized kernels -> custom ops.
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from executorch.extension.pybindings.portable_lib import _load_for_executorch  # noqa: F401
from executorch.kernels import quantized  # noqa: F401

try:
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
except Exception:
    pass

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
AUDIO_FILE = str(MODELS_DIR / "silero-vad-xnnpack" / "poem.wav")


def load_audio_wav(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load a WAV file and return float32 samples at target_sr."""
    import struct
    with open(path, "rb") as f:
        riff = f.read(4)
        assert riff == b"RIFF", f"Not a WAV file: {path}"
        f.read(4)  # chunk size
        wave = f.read(4)
        assert wave == b"WAVE"

        audio_format = 1
        num_channels = 1
        sample_rate = 16000
        bits_per_sample = 16
        data_bytes = b""

        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = f.read(chunk_size)
                audio_format = struct.unpack("<H", fmt_data[0:2])[0]
                num_channels = struct.unpack("<H", fmt_data[2:4])[0]
                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
            elif chunk_id == b"data":
                data_bytes = f.read(chunk_size)
            else:
                f.read(chunk_size)

    if bits_per_sample == 16:
        samples = struct.unpack(f"<{len(data_bytes)//2}h", data_bytes)
        audio = torch.tensor(samples, dtype=torch.float32) / 32768.0
    elif bits_per_sample == 32 and audio_format == 3:  # float32
        samples = struct.unpack(f"<{len(data_bytes)//4}f", data_bytes)
        audio = torch.tensor(samples, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported WAV format: {bits_per_sample}bit, fmt={audio_format}")

    if num_channels > 1:
        audio = audio.view(-1, num_channels).mean(dim=1)

    return audio
