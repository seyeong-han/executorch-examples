# Provenance

## Ownership and source snapshots

This package is original product source maintained in the canonical
[`meta-pytorch/executorch-examples`](https://github.com/meta-pytorch/executorch-examples)
repository under `muse_glimmer/macos/packages/livekit-plugins-executorch` and
licensed under BSD-3-Clause. The source snapshot used for the macOS subtree
migration is `914fb816fe9e0f6b7fc808fd843eb2e97df31dcf`.

The package implements adapters against the public
[`livekit/agents`](https://github.com/livekit/agents) plugin APIs at commit
`bc5f3df3a2bd1b3b8c5d1df742be57b063374991`. This ExecuTorch plugin subtree did
not exist in that upstream commit, and no LiveKit implementation source was
copied into it.

## Original source

The original product files were reorganized under this package:

- `livekit/plugins/executorch/__init__.py`
- `livekit/plugins/executorch/_helper_process.py`
- `livekit/plugins/executorch/log.py`
- `livekit/plugins/executorch/py.typed`
- `livekit/plugins/executorch/stt.py`
- `livekit/plugins/executorch/supertonic_tts.py`
- `livekit/plugins/executorch/version.py`
- `tests/fake_helper.py`
- `tests/fake_supertonic_runner.py`
- `tests/test_helper_process.py`
- `tests/test_stt.py`
- `tests/test_supertonic_tts.py`

The persistent Supertonic adapter and fake-runner tests are product-owned
implementations of the native runner's strict `--server_jsonl` protocol.

## Exclusions

This source package does not include LiveKit or ExecuTorch implementation
source, native runners, model weights, exported programs, tokenizers, voice
styles, recordings, generated output, dependency source, or build and test
caches. Those components retain their independent upstream licenses and
notices.
