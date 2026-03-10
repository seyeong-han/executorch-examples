"""Test Parakeet TDT via ExecuTorch pybindings.

Loads model.pte (contains preprocessor, encoder, decoder_step, joint methods),
runs TDT greedy decode loop with LSTM state management, decodes via SentencePiece.
"""
import time
import torch
from executorch.extension.pybindings.portable_lib import _load_for_executorch
from conftest import MODELS_DIR, load_audio_wav, AUDIO_FILE

MODEL_DIR = MODELS_DIR / "parakeet-tdt-xnnpack"
MODEL_PATH = str(MODEL_DIR / "model.pte")
TOKENIZER_PATH = str(MODEL_DIR / "tokenizer.model")

DURATIONS = [0, 1, 2, 3, 4]
MAX_SYMBOLS_PER_STEP = 10


def load_sentencepiece(path: str):
    from sentencepiece import SentencePieceProcessor
    sp = SentencePieceProcessor()
    sp.Load(path)
    return sp


def run_parakeet(audio: torch.Tensor):
    model = _load_for_executorch(MODEL_PATH)
    print(f"Methods: {model.method_names()}")

    num_rnn_layers = model.run_method("num_rnn_layers", [])[0]
    pred_hidden = model.run_method("pred_hidden", [])[0]
    vocab_size = model.run_method("vocab_size", [])[0]
    blank_id = model.run_method("blank_id", [])[0]
    sample_rate = model.run_method("sample_rate", [])[0]
    window_stride = model.run_method("window_stride", [])[0]
    enc_sub = model.run_method("encoder_subsampling_factor", [])[0]

    print(f"Metadata: vocab={vocab_size}, blank={blank_id}, "
          f"rnn_layers={num_rnn_layers}, hidden={pred_hidden}")

    t_start = time.perf_counter()

    audio_len = torch.tensor([audio.shape[0]], dtype=torch.long)
    mel_out = model.run_method("preprocessor", [audio, audio_len])
    mel = mel_out[0]
    mel_len_val = mel_out[1].item()
    print(f"Mel shape: {mel.shape}, mel_len: {mel_len_val}")

    enc_out = model.run_method("encoder", [mel, torch.tensor([mel_len_val], dtype=torch.long)])
    f_proj = enc_out[0]
    enc_len = f_proj.shape[1]
    proj_dim = f_proj.shape[2]
    print(f"Encoder output: {f_proj.shape}, enc_len: {enc_len}")

    h = torch.zeros(num_rnn_layers, 1, pred_hidden, dtype=torch.float32)
    c = torch.zeros(num_rnn_layers, 1, pred_hidden, dtype=torch.float32)

    sos_token = torch.tensor([[blank_id]], dtype=torch.long)
    dec_out = model.run_method("decoder_step", [sos_token, h, c])
    g_proj = dec_out[0].clone()
    h = dec_out[1].clone()
    c = dec_out[2].clone()

    tokens = []
    t = 0
    symbols_on_frame = 0

    while t < enc_len:
        f_t = f_proj[:, t:t+1, :].contiguous()

        joint_out = model.run_method("joint", [f_t, g_proj])
        k = joint_out[0].item()
        dur_idx = joint_out[1].item()
        dur = DURATIONS[dur_idx] if dur_idx < len(DURATIONS) else 1

        if k == blank_id:
            t += max(dur, 1)
            symbols_on_frame = 0
        else:
            tokens.append({"id": k, "offset": t, "duration": dur})

            token_input = torch.tensor([[k]], dtype=torch.long)
            dec_out = model.run_method("decoder_step", [token_input, h, c])
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

    sp = load_sentencepiece(TOKENIZER_PATH)
    token_ids = [tok["id"] for tok in tokens]
    text = sp.Decode(token_ids)

    frame_to_sec = window_stride * enc_sub
    segments = []
    current_seg_tokens = []
    for tok in tokens:
        current_seg_tokens.append(tok)
        piece = sp.IdToPiece(tok["id"])
        if piece.endswith(".") or piece.endswith("?") or piece.endswith("!"):
            start = current_seg_tokens[0]["offset"] * frame_to_sec
            end_tok = current_seg_tokens[-1]
            end_time = (end_tok["offset"] + max(end_tok["duration"], 1)) * frame_to_sec
            seg_text = sp.Decode([t["id"] for t in current_seg_tokens])
            segments.append({"start": start, "end": end_time, "text": seg_text})
            current_seg_tokens = []

    if current_seg_tokens:
        start = current_seg_tokens[0]["offset"] * frame_to_sec
        end_tok = current_seg_tokens[-1]
        end_time = (end_tok["offset"] + max(end_tok["duration"], 1)) * frame_to_sec
        seg_text = sp.Decode([t["id"] for t in current_seg_tokens])
        segments.append({"start": start, "end": end_time, "text": seg_text})

    return text, tokens, segments, elapsed


def main():
    print(f"Loading audio: {AUDIO_FILE}")
    audio = load_audio_wav(AUDIO_FILE)
    print(f"Audio: {audio.shape[0]} samples, {audio.shape[0]/16000:.1f}s")

    print(f"\nLoading model: {MODEL_PATH}")
    text, tokens, segments, elapsed = run_parakeet(audio)

    print(f"\nTranscription:\n{text}\n")
    print(f"Tokens: {len(tokens)}")
    print(f"Segments ({len(segments)}):")
    for seg in segments:
        print(f"  {seg['start']:.2f}s - {seg['end']:.2f}s : {seg['text']}")
    print(f"\nInference time: {elapsed:.3f}s")
    print(f"\nParakeet TDT test PASSED")


if __name__ == "__main__":
    main()
