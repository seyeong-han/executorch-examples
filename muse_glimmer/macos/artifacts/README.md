# Local artifacts

This directory contains manifests only. Models, complete tokenizer bundles,
voice styles, exported programs, and native binaries live under ignored
`.local/artifacts/`.

Each artifact is governed by its own upstream license. The BSD-3-Clause
license for product-owned source does not apply to those artifacts. Run
`make prepare-artifacts` after reviewing the licenses and providing any
artifacts marked `user-provided` in `macos-arm64.lock.json`.

Preparation verifies checksums and writes `.local/state/prepared.json`. The
inventory includes the shared `mlx.metallib` that must be colocated with all
three native executables under `.local/artifacts/bin/`. Daily startup consumes
that receipt and never downloads, builds, or exports assets.
