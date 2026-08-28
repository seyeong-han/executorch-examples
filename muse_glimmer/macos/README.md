# Muse Glimmer Voice Agent

A fully local voice agent for macOS on Apple silicon. The browser captures the
microphone, loopback-only LiveKit carries audio, and local ExecuTorch runtimes
perform Parakeet speech recognition, Muse Glimmer generation, and Supertonic
speech synthesis.

```text
browser microphone
  -> 127.0.0.1 LiveKit
  -> Parakeet ASR
  -> Muse Glimmer LLM
  -> Supertonic TTS
  -> browser speaker
```

No LiveKit Cloud account or cloud inference service is used.

## Status

The native compatibility pin is ExecuTorch
`20ad5ee43ff53804030899d621590af3daadda53`, which contains the landed
Supertonic runtime, bounded MuseGlimmer worker cancellation, and persistent
Supertonic JSONL mode. Release readiness remains false until final artifact
provenance and clean-machine macOS arm64 end-to-end validation are complete.
See `docs/upstream-pins.md`.

## Supported platform

- macOS on Apple silicon
- Python 3.13
- Node.js 22
- A compatible Xcode/CMake toolchain
- LiveKit Server 1.x

Other platforms are not part of the first milestone.

## First-time setup

Run application commands from the subtree root:

```bash
cd muse_glimmer/macos
```

Review the independent model and runtime licenses before providing artifacts.
Models and native binaries are stored only under ignored `.local/` paths. Source
checks and package builds are available now:

```bash
make check
make test
```

With a clean ExecuTorch checkout at the locked commit, run:

```bash
make bootstrap
make prepare-artifacts
```

`make bootstrap` validates the locked toolchain and installs source
dependencies. `make prepare-artifacts` validates the single pinned ExecuTorch
checkout and every model/native artifact, then writes an ignored compatibility
receipt. It does not download, build, export, or repair missing artifacts.
Neither operation runs during normal startup.

## Daily development

```bash
make dev
make status
make logs
make restart
make down
```

`make dev up` is also supported and starts the stack exactly once. Once
artifacts have been prepared, startup requires no external network access.

The UI opens at `http://127.0.0.1:5173`.

## Local security boundary

- LiveKit signaling: `127.0.0.1:7880`
- LiveKit media: `127.0.0.1:7882/udp`
- Token service: `127.0.0.1:8787`
- Browser UI: `127.0.0.1:5173`
- MuseGlimmer server: backend-only `127.0.0.1:8000`
- Short-lived participant tokens grant microphone publication only.
- Runtime LiveKit credentials are generated locally per stack run.
- Browser code never receives model identifiers, artifact paths, the LLM
  endpoint, native worker details, or server credentials.

See `docs/security-model.md` for the local-process trust model.

## Development checks

From `muse_glimmer/macos`:

```bash
make check
make test
make publication-check
```

The publication check rejects secrets, models, native binaries, recordings,
generated output, nested repositories, absolute workstation paths, internal
URLs, and AGPL avatar dependencies.

## License

Product-owned source is BSD-3-Clause. Models, exported programs, native
binaries, fonts, and third-party packages retain their independent licenses.
See `LICENSE`, `PROVENANCE.md`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/`.
