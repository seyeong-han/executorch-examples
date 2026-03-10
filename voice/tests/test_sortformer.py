"""Test Sortformer via ExecuTorch pybindings.

Loads sortformer.pte, runs the 3-stage diarization pipeline:
preprocessor -> pre_encode -> streaming encode with FIFO + speaker cache.
"""
import time
import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from conftest import MODELS_DIR, load_audio_wav, AUDIO_FILE

MODEL_DIR = MODELS_DIR / "sortformer-xnnpack"
MODEL_PATH = str(MODEL_DIR / "sortformer.pte")

MEL_BINS = 128
MAX_MEL_FRAMES = 4000
THRESHOLD = 0.5
CHUNK_LEN = 124
FIFO_LEN = 124


def compress_cache(embs, preds, cache_size, max_size, d_model, max_spks):
    if cache_size <= max_size:
        return embs, preds, cache_size
    scored = []
    for i in range(cache_size):
        max_p = max(preds[i * max_spks + s] for s in range(max_spks))
        scored.append((max_p, i))
    scored.sort(key=lambda x: -x[0])
    keep = sorted([scored[i][1] for i in range(max_size)])
    new_embs = []
    new_preds = []
    for i in keep:
        new_embs.extend(embs[i * d_model : (i + 1) * d_model])
        new_preds.extend(preds[i * max_spks : (i + 1) * max_spks])
    return new_embs, new_preds, max_size


def run_sortformer(audio: torch.Tensor):
    model = _load_for_executorch(MODEL_PATH)
    print(f"Methods: {model.method_names()}")

    window_stride = model.run_method("window_stride", [])[0]
    sub_factor = model.run_method("subsampling_factor", [])[0]
    spkcache_len = model.run_method("spkcache_len", [])[0]
    max_spks = model.run_method("max_num_of_spks", [])[0]
    frame_duration = window_stride * sub_factor

    print(f"Metadata: max_spks={max_spks}, spkcache_len={spkcache_len}, "
          f"frame_duration={frame_duration:.3f}s")

    t_start = time.perf_counter()

    # Stage 1: preprocessor
    audio_len = torch.tensor([audio.shape[0]], dtype=torch.long)
    preproc_out = model.run_method("preprocessor", [audio, audio_len])
    mel = preproc_out[0]  # [1, 128, T_mel]
    mel_len_val = preproc_out[1].item()
    valid_mel = min(mel.shape[2], mel_len_val)
    print(f"Mel shape: {mel.shape}, valid_mel: {valid_mel}")

    # Transpose: [1, 128, T] -> [T, 128] (channels-first to channels-last)
    mel_np = mel[0, :, :valid_mel].T.contiguous()  # [T, 128]

    # Stage 2: pre_encode (pad to MAX_MEL_FRAMES)
    all_embs = []
    total_emb_len = 0
    d_model = 0

    for mel_offset in range(0, valid_mel, MAX_MEL_FRAMES):
        sub_len = min(MAX_MEL_FRAMES, valid_mel - mel_offset)
        padded = torch.zeros(1, MAX_MEL_FRAMES, MEL_BINS, dtype=torch.float32)
        padded[0, :sub_len, :] = mel_np[mel_offset:mel_offset + sub_len, :]
        sub_len_t = torch.tensor([sub_len], dtype=torch.long)

        enc_out = model.run_method("pre_encode", [padded, sub_len_t])
        embs = enc_out[0]  # [1, emb_len, d_model]
        emb_len = enc_out[1].item()

        if d_model == 0:
            d_model = embs.shape[2]

        embs_flat = embs[0, :emb_len, :].contiguous().view(-1).tolist()
        all_embs.extend(embs_flat)
        total_emb_len += emb_len

    print(f"Total embeddings: {total_emb_len} frames x {d_model} dims")

    # Stage 3: streaming encode with FIFO + speaker cache
    cache_embs = []
    cache_preds = []
    cache_size = 0
    fifo_embs = []
    fifo_size = 0

    spk_active = [False] * max_spks
    spk_start_frame = [0] * max_spks
    spk_active_frames = [0] * max_spks
    num_output_frames = 0
    segments = []

    for offset in range(0, total_emb_len, CHUNK_LEN):
        cur_chunk_len = min(CHUNK_LEN, total_emb_len - offset)
        total_len = cache_size + fifo_size + cur_chunk_len

        concat = []
        if cache_size > 0:
            concat.extend(cache_embs)
        if fifo_size > 0:
            concat.extend(fifo_embs)
        chunk_start = offset * d_model
        concat.extend(all_embs[chunk_start : chunk_start + cur_chunk_len * d_model])

        enc_tensor = torch.tensor(concat, dtype=torch.float32).view(1, total_len, d_model)
        enc_len_tensor = torch.tensor([total_len], dtype=torch.long)

        enc_result = model.run_method("encode", [enc_tensor, enc_len_tensor])
        preds = enc_result[0]  # [1, total_len, max_spks]
        preds_flat = preds[0].contiguous().view(-1).tolist()

        if cache_size > 0:
            cache_preds = preds_flat[:cache_size * max_spks]

        chunk_pred_start = (cache_size + fifo_size) * max_spks
        for t in range(cur_chunk_len):
            global_frame = num_output_frames + t
            for spk in range(max_spks):
                prob = preds_flat[chunk_pred_start + t * max_spks + spk]
                if prob > THRESHOLD:
                    spk_active_frames[spk] += 1
                    if not spk_active[spk]:
                        spk_active[spk] = True
                        spk_start_frame[spk] = global_frame
                elif spk_active[spk]:
                    spk_active[spk] = False
                    segments.append({
                        "start": spk_start_frame[spk] * frame_duration,
                        "end": global_frame * frame_duration,
                        "speaker": f"speaker_{spk}",
                    })
        num_output_frames += cur_chunk_len

        # Update FIFO
        combined = fifo_size + cur_chunk_len
        overflow = max(0, combined - FIFO_LEN)
        if overflow > 0:
            from_fifo = min(overflow, fifo_size)
            from_chunk = overflow - from_fifo
            if from_fifo > 0:
                cache_embs.extend(fifo_embs[:from_fifo * d_model])
                fifo_pred_start = cache_size * max_spks
                cache_preds.extend(
                    preds_flat[fifo_pred_start : fifo_pred_start + from_fifo * max_spks]
                )
                cache_size += from_fifo
                fifo_embs = fifo_embs[from_fifo * d_model:]
                fifo_size -= from_fifo
            if from_chunk > 0:
                cs = offset * d_model
                cache_embs.extend(all_embs[cs : cs + from_chunk * d_model])
                cache_preds.extend(
                    preds_flat[chunk_pred_start : chunk_pred_start + from_chunk * max_spks]
                )
                cache_size += from_chunk
            chunk_keep = cur_chunk_len - from_chunk
            if chunk_keep > 0:
                ks = (offset + from_chunk) * d_model
                fifo_embs.extend(all_embs[ks : ks + chunk_keep * d_model])
                fifo_size += chunk_keep
        else:
            es = offset * d_model
            fifo_embs.extend(all_embs[es : es + cur_chunk_len * d_model])
            fifo_size += cur_chunk_len

        if cache_size > spkcache_len:
            cache_embs, cache_preds, cache_size = compress_cache(
                cache_embs, cache_preds, cache_size, spkcache_len, d_model, max_spks
            )

    for spk in range(max_spks):
        if spk_active[spk]:
            segments.append({
                "start": spk_start_frame[spk] * frame_duration,
                "end": num_output_frames * frame_duration,
                "speaker": f"speaker_{spk}",
            })

    elapsed = time.perf_counter() - t_start
    return segments, num_output_frames, spk_active_frames, elapsed


def main():
    print(f"Loading audio: {AUDIO_FILE}")
    audio = load_audio_wav(AUDIO_FILE)
    print(f"Audio: {audio.shape[0]} samples, {audio.shape[0]/16000:.1f}s")

    print(f"\nLoading model: {MODEL_PATH}")
    segments, total_frames, spk_frames, elapsed = run_sortformer(audio)

    print(f"\nSpeaker segments ({len(segments)}):")
    for seg in segments:
        print(f"  {seg['start']:.3f}s - {seg['end']:.3f}s {seg['speaker']}")

    print(f"\nTotal frames: {total_frames}")
    for i, frames in enumerate(spk_frames):
        if frames > 0:
            print(f"  Speaker {i}: {frames}/{total_frames} frames ({frames/total_frames:.1%})")
    print(f"Inference time: {elapsed:.3f}s")
    print(f"\nSortformer test PASSED")


if __name__ == "__main__":
    main()
