# Upstream Compatibility Pins

The application source can be developed in this repository, but the native
compatibility lock remains `development-gated` until one immutable ExecuTorch
commit contains every required interface.

## Capability status

- ExecuTorch PR [#22063](https://github.com/pytorch/executorch/pull/22063):
  base Supertonic export and native runtime landed at
  `81969a92dd2e5515fa23ccdf9d87346cf3ba2ba2`.
- ExecuTorch PR [#22070](https://github.com/pytorch/executorch/pull/22070):
  bounded generic LLM worker cancellation and health propagation are pending
  merge. The application will record the public landed commit after merge.
- Persistent Supertonic JSONL mode: the follow-up is not submitted yet. Its
  protocol-v1 ready frame must report `sample_rate: 44100` together with load
  and warmup timing so the Python adapter and native runtime enforce one schema.

The merged base Supertonic runtime alone is not a release pin. A temporary
public PR commit may be used for local development only when it is pinned by
full SHA, publicly accessible, license-compatible, and validated as a single
checkout. Branch names, dirty checkouts, and mixed Python/native revisions are
forbidden.

Before setting `ready_for_release` to true:

1. Land every required capability and select one descendant ExecuTorch commit.
2. Populate artifact sources, revisions, checksums, sizes, and exact licenses.
3. Run real generation, stream cancellation, and post-cancel generation.
4. Verify multiple Supertonic utterances reuse one warm process using the
   documented protocol-v1 ready frame.
5. Pass a clean-machine macOS arm64 end-to-end run.
