import os
import platform
import shutil
import subprocess
from pathlib import Path

VOICE_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = VOICE_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

EXECUTORCH_ROOT = Path(os.environ.get(
    "EXECUTORCH_ROOT",
    Path.home() / "executorch",
))
CMAKE_OUT = EXECUTORCH_ROOT / "cmake-out" / "examples" / "models"

SYSTEM = platform.system().lower()
IS_MAC = SYSTEM == "darwin"
IS_LINUX = SYSTEM == "linux"
IS_WINDOWS = SYSTEM == "windows"


def _has_nvidia_gpu() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi"], capture_output=True, check=True, timeout=5,
        )
        return True
    except Exception:
        return False


HAS_CUDA = _has_nvidia_gpu() and not IS_MAC

AVAILABLE_BACKENDS: list[str] = ["xnnpack"]
if IS_MAC:
    AVAILABLE_BACKENDS.append("metal")
if HAS_CUDA:
    AVAILABLE_BACKENDS.append("cuda")

LIBOMP_PATH: str | None = None
if IS_MAC:
    brew_prefix = shutil.which("brew")
    if brew_prefix:
        try:
            result = subprocess.run(
                ["brew", "--prefix", "libomp"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                LIBOMP_PATH = result.stdout.strip() + "/lib"
        except Exception:
            pass


RUNNER_BINARIES = {
    "whisper": "whisper/whisper_runner",
    "parakeet-tdt": "parakeet/parakeet_runner",
    "voxtral-realtime": "voxtral_realtime/voxtral_realtime_runner",
    "sortformer": "sortformer/sortformer_runner",
    "silero-vad": "silero_vad/silero_vad_runner",
}

HF_MODEL_CATALOG = {
    "whisper-tiny": {
        "task": "transcription",
        "runner": "whisper",
        "display_name": "Whisper Tiny",
        "params": "39M",
        "repos": {
            "xnnpack": "younghan-meta/Whisper-Tiny-ExecuTorch-XNNPACK",
            "metal": "younghan-meta/Whisper-Tiny-ExecuTorch-Metal",
        },
    },
    "parakeet-tdt": {
        "task": "transcription",
        "runner": "parakeet-tdt",
        "display_name": "Parakeet TDT 0.6B",
        "params": "0.6B",
        "repos": {
            "xnnpack": "younghan-meta/Parakeet-TDT-ExecuTorch-XNNPACK",
            "metal": "younghan-meta/Parakeet-TDT-ExecuTorch-Metal",
        },
    },
    "voxtral-realtime": {
        "task": "transcription",
        "runner": "voxtral-realtime",
        "display_name": "Voxtral Realtime 4B",
        "params": "4B",
        "repos": {
            "xnnpack": "younghan-meta/Voxtral-Mini-4B-Realtime-2602-ExecuTorch-XNNPACK",
            "metal": "younghan-meta/Voxtral-Mini-4B-Realtime-2602-ExecuTorch-Metal",
        },
    },
    "sortformer": {
        "task": "diarization",
        "runner": "sortformer",
        "display_name": "Sortformer 4-Speaker",
        "params": "117M",
        "repos": {
            "xnnpack": "younghan-meta/Sortformer-ExecuTorch-XNNPACK",
        },
    },
    "silero-vad": {
        "task": "vad",
        "runner": "silero-vad",
        "display_name": "Silero VAD",
        "params": "2MB",
        "repos": {
            "xnnpack": "younghan-meta/Silero-VAD-ExecuTorch-XNNPACK",
        },
    },
}

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
