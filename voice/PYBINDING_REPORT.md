# ExecuTorch Voice Studio: Python Runtime Architecture

## Summary

The Voice Studio backend runs all 5 voice models through a unified Python process. Four models use ExecuTorch pybindings for direct in-process inference. One model (Voxtral) uses a subprocess C++ runner due to a confirmed KV cache bug in the pybindings layer.

## Model → Runtime Mapping

| Model | Runtime | Verified | Perf |
|-------|---------|----------|------|
| Silero VAD | Pybinding | 8 segments, 80.5% speech | 0.30s |
| Whisper Tiny | Pybinding | Correct transcription | 10.2s, 7.8 tok/s |
| Parakeet TDT | Pybinding | Correct text + timestamps | 4.5s, 24 tok/s |
| Sortformer | Pybinding | 2 speakers detected | 1.9s |
| Voxtral Realtime | C++ subprocess | Correct transcription | 31.5s, 17.5 tok/s |

No fallback logic. Each model has exactly one runtime path.

## Pybinding Architecture

```
FastAPI request
  → _resolve_model() selects decoder based on model ID
  → Pybinding decoder loads .pte via _load_for_executorch()
  → Decoder runs model methods (encoder, text_decoder, etc.)
  → Returns structured Python result objects
```

Backend auto-detection is built into ExecuTorch: the `.pte` file encodes its delegate ID (`XnnpackBackend`, `MetalBackend`, `CudaBackend`). The pybindings dispatch automatically — same Python code runs XNNPACK, Metal, or CUDA depending on the `.pte` file loaded.

## Voxtral KV Cache Bug

**Root cause confirmed:** The Voxtral model's `text_decoder` KV cache does not persist between `run_method()` calls via pybindings. Calling `text_decoder` at `cache_position=0` then `cache_position=1` with identical input produces identical output (diff=0.000004, effectively zero).

This was verified by comparing against:
- **Whisper's KV cache** — works correctly via the same pybindings (diff=25.7 between positions)
- **Voxtral's C++ runner** — works correctly with the same `.pte` file

The bug is specific to how Voxtral's XNNPACK-delegated model handles mutable buffer writes in the SDPA layer. The model was exported via `export_voxtral_rt.py` with `custom_sdpa_with_kv_cache`, while Whisper was exported via `optimum-executorch`. The different export paths produce different KV cache buffer management code.

**Status:** Voxtral uses the C++ runner (subprocess) until this export-level bug is fixed. No workaround or fallback — this is the designated runtime for Voxtral.

## Files

```
voice/backend/
├── decoders/                        # Pybinding decoders (4 models)
│   ├── __init__.py                  # Op registration
│   ├── silero_vad_decoder.py        # forward loop + LSTM state
│   ├── whisper_decoder.py           # encoder + autoregressive decoder
│   ├── parakeet_decoder.py          # TDT transducer greedy decode
│   └── sortformer_decoder.py        # 3-stage pipeline + FIFO/cache
├── runners/
│   ├── base.py                      # Subprocess runner base
│   └── voxtral.py                   # Voxtral subprocess runner
├── routers/
│   └── audio.py                     # Routes to decoder or runner by model ID
voice/tests/                         # Standalone test scripts per model
```
