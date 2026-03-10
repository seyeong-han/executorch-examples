from __future__ import annotations

import time
import torch
from pathlib import Path
from executorch.extension.pybindings.portable_lib import _load_for_executorch

from . import ensure_ops_registered
from ..runners.base import PerformanceStats, TranscriptionResult

DURATIONS = [0, 1, 2, 3, 4]
MAX_SYMBOLS_PER_STEP = 10


class ParakeetDecoder:
    def __init__(self, model_dir: Path):
        ensure_ops_registered()
        self.model = _load_for_executorch(str(model_dir / "model.pte"))
        self.tokenizer_path = str(model_dir / "tokenizer.model")

        self.num_rnn_layers = self.model.run_method("num_rnn_layers", [])[0]
        self.pred_hidden = self.model.run_method("pred_hidden", [])[0]
        self.blank_id = self.model.run_method("blank_id", [])[0]
        self.window_stride = self.model.run_method("window_stride", [])[0]
        self.enc_sub = self.model.run_method("encoder_subsampling_factor", [])[0]

    def run(self, audio: torch.Tensor, timestamps: str = "segment") -> TranscriptionResult:
        t_start = time.perf_counter()

        audio_len = torch.tensor([audio.shape[0]], dtype=torch.long)
        mel_out = self.model.run_method("preprocessor", [audio, audio_len])
        mel = mel_out[0]
        mel_len_val = mel_out[1].item()

        enc_out = self.model.run_method("encoder", [mel, torch.tensor([mel_len_val], dtype=torch.long)])
        f_proj = enc_out[0]
        enc_len = f_proj.shape[1]

        h = torch.zeros(self.num_rnn_layers, 1, self.pred_hidden, dtype=torch.float32)
        c = torch.zeros(self.num_rnn_layers, 1, self.pred_hidden, dtype=torch.float32)

        sos_token = torch.tensor([[self.blank_id]], dtype=torch.long)
        dec_out = self.model.run_method("decoder_step", [sos_token, h, c])
        g_proj = dec_out[0].clone()
        h = dec_out[1].clone()
        c = dec_out[2].clone()

        tokens = []
        t = 0
        symbols_on_frame = 0

        while t < enc_len:
            f_t = f_proj[:, t:t+1, :].contiguous()
            joint_out = self.model.run_method("joint", [f_t, g_proj])
            k = joint_out[0].item()
            dur_idx = joint_out[1].item()
            dur = DURATIONS[dur_idx] if dur_idx < len(DURATIONS) else 1

            if k == self.blank_id:
                t += max(dur, 1)
                symbols_on_frame = 0
            else:
                tokens.append({"id": k, "offset": t, "duration": dur})
                token_input = torch.tensor([[k]], dtype=torch.long)
                dec_out = self.model.run_method("decoder_step", [token_input, h, c])
                g_proj = dec_out[0].clone()
                h = dec_out[1].clone()
                c = dec_out[2].clone()
                t += dur
                if dur == 0:
                    symbols_on_frame += 1
                    if symbols_on_frame >= MAX_SYMBOLS_PER_STEP:
                        t += 1
                        symbols_on_frame = 0
                else:
                    symbols_on_frame = 0

        elapsed = time.perf_counter() - t_start

        from sentencepiece import SentencePieceProcessor
        sp = SentencePieceProcessor()
        sp.Load(self.tokenizer_path)

        token_ids = [tok["id"] for tok in tokens]
        text = sp.Decode(token_ids)

        segments = []
        if timestamps != "none":
            frame_to_sec = self.window_stride * self.enc_sub
            current_seg = []
            for tok in tokens:
                current_seg.append(tok)
                piece = sp.IdToPiece(tok["id"])
                if piece.endswith((".","?","!")):
                    start = current_seg[0]["offset"] * frame_to_sec
                    end_t = current_seg[-1]
                    end_time = (end_t["offset"] + max(end_t["duration"], 1)) * frame_to_sec
                    seg_text = sp.Decode([t["id"] for t in current_seg])
                    segments.append({"start": start, "end": end_time, "text": seg_text})
                    current_seg = []
            if current_seg:
                start = current_seg[0]["offset"] * frame_to_sec
                end_t = current_seg[-1]
                end_time = (end_t["offset"] + max(end_t["duration"], 1)) * frame_to_sec
                seg_text = sp.Decode([t["id"] for t in current_seg])
                segments.append({"start": start, "end": end_time, "text": seg_text})

        return TranscriptionResult(
            text=text,
            segments=segments,
            performance=PerformanceStats(
                inference_time_s=elapsed,
                tokens_generated=len(tokens),
                tokens_per_second=len(tokens) / elapsed if elapsed > 0 else 0,
            ),
        )
