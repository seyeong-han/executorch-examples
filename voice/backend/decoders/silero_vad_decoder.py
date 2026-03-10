from __future__ import annotations

import time
import torch
from pathlib import Path
from executorch.extension.pybindings.portable_lib import _load_for_executorch

from . import ensure_ops_registered
from ..runners.base import PerformanceStats, VadResult

WINDOW_SIZE = 512
CONTEXT_SIZE = 64
INPUT_SIZE = WINDOW_SIZE + CONTEXT_SIZE
HIDDEN_DIM = 128
SAMPLE_RATE = 16000


class SileroVadDecoder:
    def __init__(self, model_dir: Path):
        ensure_ops_registered()
        self.module = _load_for_executorch(str(model_dir / "silero_vad.pte"))

    def run(self, audio: torch.Tensor, threshold: float = 0.5) -> VadResult:
        state = torch.zeros(2, 1, HIDDEN_DIM, dtype=torch.float32)
        context = torch.zeros(CONTEXT_SIZE, dtype=torch.float32)

        num_samples = audio.shape[0]
        segments = []
        speech_active = False
        speech_start = 0.0
        speech_frames = 0
        total_frames = 0

        t_start = time.perf_counter()
        offset = 0
        while offset + WINDOW_SIZE <= num_samples:
            chunk = audio[offset : offset + WINDOW_SIZE]
            x = torch.cat([context, chunk]).unsqueeze(0)

            outputs = self.module.run_method("forward", [x, state])
            prob = outputs[0]
            state = outputs[1]

            prob_val = prob.item()
            frame_time = offset / SAMPLE_RATE
            total_frames += 1

            if prob_val > threshold:
                speech_frames += 1
                if not speech_active:
                    speech_active = True
                    speech_start = frame_time
            elif speech_active:
                speech_active = False
                segments.append({"start": speech_start, "end": frame_time})

            context = chunk[-CONTEXT_SIZE:]
            offset += WINDOW_SIZE

        if speech_active:
            segments.append({"start": speech_start, "end": num_samples / SAMPLE_RATE})

        elapsed = time.perf_counter() - t_start
        speech_ratio = speech_frames / max(total_frames, 1)

        return VadResult(
            segments=segments,
            speech_ratio=speech_ratio,
            performance=PerformanceStats(inference_time_s=elapsed),
        )
