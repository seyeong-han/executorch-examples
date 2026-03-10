"""Test Silero VAD via ExecuTorch pybindings.

Loads silero_vad.pte, processes audio in 512-sample chunks with 64-sample
context overlap, carries LSTM state across chunks, and extracts speech segments.
"""
import time
import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from conftest import MODELS_DIR, load_audio_wav, AUDIO_FILE

MODEL_DIR = MODELS_DIR / "silero-vad-xnnpack"
MODEL_PATH = str(MODEL_DIR / "silero_vad.pte")

WINDOW_SIZE = 512
CONTEXT_SIZE = 64
INPUT_SIZE = WINDOW_SIZE + CONTEXT_SIZE  # 576
HIDDEN_DIM = 128
SAMPLE_RATE = 16000
THRESHOLD = 0.5


def run_vad(audio: torch.Tensor, threshold: float = THRESHOLD):
    module = _load_for_executorch(MODEL_PATH)

    print(f"Methods: {module.method_names()}")

    state = torch.zeros(2, 1, HIDDEN_DIM, dtype=torch.float32)
    context = torch.zeros(CONTEXT_SIZE, dtype=torch.float32)

    num_samples = audio.shape[0]
    frame_duration = WINDOW_SIZE / SAMPLE_RATE
    segments = []
    speech_active = False
    speech_start = 0.0
    speech_frames = 0
    total_frames = 0

    t_start = time.perf_counter()

    offset = 0
    while offset + WINDOW_SIZE <= num_samples:
        chunk = audio[offset : offset + WINDOW_SIZE]
        x = torch.cat([context, chunk]).unsqueeze(0)  # [1, 576]

        outputs = module.run_method("forward", [x, state])
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
        else:
            if speech_active:
                speech_active = False
                segments.append({"start": speech_start, "end": frame_time})

        context = chunk[-CONTEXT_SIZE:]
        offset += WINDOW_SIZE

    if speech_active:
        segments.append({"start": speech_start, "end": num_samples / SAMPLE_RATE})

    elapsed = time.perf_counter() - t_start
    speech_ratio = speech_frames / max(total_frames, 1)

    return segments, speech_ratio, elapsed, total_frames


def main():
    print(f"Loading audio: {AUDIO_FILE}")
    audio = load_audio_wav(AUDIO_FILE)
    print(f"Audio: {audio.shape[0]} samples, {audio.shape[0]/SAMPLE_RATE:.1f}s")

    print(f"\nLoading model: {MODEL_PATH}")
    segments, speech_ratio, elapsed, total_frames = run_vad(audio)

    print(f"\nSpeech segments ({len(segments)}):")
    for seg in segments:
        print(f"  {seg['start']:.3f}s - {seg['end']:.3f}s")

    print(f"\nSpeech ratio: {speech_ratio:.1%}")
    print(f"Frames: {total_frames}")
    print(f"Inference time: {elapsed:.3f}s")
    print(f"\nSilero VAD test PASSED")


if __name__ == "__main__":
    main()
