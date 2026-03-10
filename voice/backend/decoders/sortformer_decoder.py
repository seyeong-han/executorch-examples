from __future__ import annotations

import time
import torch
from pathlib import Path
from executorch.extension.pybindings.portable_lib import _load_for_executorch

from . import ensure_ops_registered
from ..runners.base import DiarizationResult, PerformanceStats

MEL_BINS = 128
MAX_MEL_FRAMES = 4000
DEFAULT_CHUNK_LEN = 124
DEFAULT_FIFO_LEN = 124


def _compress_cache(embs, preds, cache_size, max_size, d_model, max_spks):
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


class SortformerDecoder:
    def __init__(self, model_dir: Path):
        ensure_ops_registered()
        self.model = _load_for_executorch(str(model_dir / "sortformer.pte"))
        self.spkcache_len = self.model.run_method("spkcache_len", [])[0]
        self.max_spks = self.model.run_method("max_num_of_spks", [])[0]
        self.window_stride = self.model.run_method("window_stride", [])[0]
        self.sub_factor = self.model.run_method("subsampling_factor", [])[0]
        self.frame_duration = self.window_stride * self.sub_factor

    def run(self, audio: torch.Tensor, threshold: float = 0.5) -> DiarizationResult:
        t_start = time.perf_counter()

        audio_len = torch.tensor([audio.shape[0]], dtype=torch.long)
        preproc_out = self.model.run_method("preprocessor", [audio, audio_len])
        mel = preproc_out[0]
        mel_len_val = preproc_out[1].item()
        valid_mel = min(mel.shape[2], mel_len_val)

        mel_transposed = mel[0, :, :valid_mel].T.contiguous()

        all_embs = []
        total_emb_len = 0
        d_model = 0

        for mel_offset in range(0, valid_mel, MAX_MEL_FRAMES):
            sub_len = min(MAX_MEL_FRAMES, valid_mel - mel_offset)
            padded = torch.zeros(1, MAX_MEL_FRAMES, MEL_BINS, dtype=torch.float32)
            padded[0, :sub_len, :] = mel_transposed[mel_offset:mel_offset + sub_len, :]

            enc_out = self.model.run_method("pre_encode", [padded, torch.tensor([sub_len], dtype=torch.long)])
            embs = enc_out[0]
            emb_len = enc_out[1].item()
            if d_model == 0:
                d_model = embs.shape[2]
            all_embs.extend(embs[0, :emb_len, :].contiguous().view(-1).tolist())
            total_emb_len += emb_len

        cache_embs, cache_preds = [], []
        cache_size = 0
        fifo_embs = []
        fifo_size = 0
        max_spks = self.max_spks
        spk_active = [False] * max_spks
        spk_start_frame = [0] * max_spks
        spk_active_frames = [0] * max_spks
        num_output_frames = 0
        segments = []

        for offset in range(0, total_emb_len, DEFAULT_CHUNK_LEN):
            cur_chunk_len = min(DEFAULT_CHUNK_LEN, total_emb_len - offset)
            total_len = cache_size + fifo_size + cur_chunk_len

            concat = []
            if cache_size > 0:
                concat.extend(cache_embs)
            if fifo_size > 0:
                concat.extend(fifo_embs)
            concat.extend(all_embs[offset * d_model : (offset + cur_chunk_len) * d_model])

            enc_tensor = torch.tensor(concat, dtype=torch.float32).view(1, total_len, d_model)
            enc_result = self.model.run_method("encode", [enc_tensor, torch.tensor([total_len], dtype=torch.long)])
            preds_flat = enc_result[0][0].contiguous().view(-1).tolist()

            if cache_size > 0:
                cache_preds = preds_flat[:cache_size * max_spks]

            chunk_pred_start = (cache_size + fifo_size) * max_spks
            for t in range(cur_chunk_len):
                global_frame = num_output_frames + t
                for spk in range(max_spks):
                    prob = preds_flat[chunk_pred_start + t * max_spks + spk]
                    if prob > threshold:
                        spk_active_frames[spk] += 1
                        if not spk_active[spk]:
                            spk_active[spk] = True
                            spk_start_frame[spk] = global_frame
                    elif spk_active[spk]:
                        spk_active[spk] = False
                        segments.append({
                            "start": spk_start_frame[spk] * self.frame_duration,
                            "end": global_frame * self.frame_duration,
                            "speaker": f"speaker_{spk}",
                        })
            num_output_frames += cur_chunk_len

            combined = fifo_size + cur_chunk_len
            overflow = max(0, combined - DEFAULT_FIFO_LEN)
            if overflow > 0:
                from_fifo = min(overflow, fifo_size)
                from_chunk = overflow - from_fifo
                if from_fifo > 0:
                    cache_embs.extend(fifo_embs[:from_fifo * d_model])
                    fp_start = cache_size * max_spks
                    cache_preds.extend(preds_flat[fp_start : fp_start + from_fifo * max_spks])
                    cache_size += from_fifo
                    fifo_embs = fifo_embs[from_fifo * d_model:]
                    fifo_size -= from_fifo
                if from_chunk > 0:
                    cs = offset * d_model
                    cache_embs.extend(all_embs[cs : cs + from_chunk * d_model])
                    cache_preds.extend(preds_flat[chunk_pred_start : chunk_pred_start + from_chunk * max_spks])
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

            if cache_size > self.spkcache_len:
                cache_embs, cache_preds, cache_size = _compress_cache(
                    cache_embs, cache_preds, cache_size, self.spkcache_len, d_model, max_spks
                )

        for spk in range(max_spks):
            if spk_active[spk]:
                segments.append({
                    "start": spk_start_frame[spk] * self.frame_duration,
                    "end": num_output_frames * self.frame_duration,
                    "speaker": f"speaker_{spk}",
                })

        elapsed = time.perf_counter() - t_start
        num_speakers = len(set(s["speaker"] for s in segments))

        return DiarizationResult(
            segments=segments,
            num_speakers=num_speakers,
            performance=PerformanceStats(inference_time_s=elapsed),
        )
