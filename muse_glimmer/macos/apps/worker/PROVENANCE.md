# Provenance

## Ownership and source snapshots

This package is original product source maintained in the canonical
[`meta-pytorch/executorch-examples`](https://github.com/meta-pytorch/executorch-examples)
repository under `muse_glimmer/macos/apps/worker` and licensed under
BSD-3-Clause. The source snapshot used for the macOS subtree migration is
`914fb816fe9e0f6b7fc808fd843eb2e97df31dcf`.

The package uses the public
[`livekit/agents`](https://github.com/livekit/agents) APIs at commit
`bc5f3df3a2bd1b3b8c5d1df742be57b063374991`. The package did not exist in that
upstream commit, and no LiveKit implementation source was copied into it.

## Source mapping

The original product files were reorganized for this standalone package:

- `agents/examples/voice_agents/glimmer_agent.py` -> `src/muse_glimmer_worker/agent.py`
- `agents/examples/voice_agents/glimmer_cli.py` -> `src/muse_glimmer_worker/cli.py`
- `agents/examples/voice_agents/glimmer_config.py` -> `src/muse_glimmer_worker/config.py`
- `agents/livekit-plugins/livekit-plugins-executorch/tests/test_glimmer_agent_privacy.py`
  -> `tests/test_agent.py`
- `agents/livekit-plugins/livekit-plugins-executorch/tests/test_glimmer_cli.py`
  -> `tests/test_cli.py`
- `agents/livekit-plugins/livekit-plugins-executorch/tests/test_glimmer_config.py`
  -> `tests/test_config.py`

Packaging and lifecycle code are original additions for this distribution.

## Exclusions

This source package does not include LiveKit or ExecuTorch implementation
source, native runners, model weights, exported programs, tokenizers, voice
styles, recordings, generated output, dependency source, or build and test
caches. Those components retain their independent upstream licenses and
notices.
