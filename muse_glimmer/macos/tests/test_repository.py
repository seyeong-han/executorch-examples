from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import repository


def test_relative_local_path_rejects_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / ".local"
    local.mkdir()
    monkeypatch.setattr(repository, "ROOT", tmp_path)
    monkeypatch.setattr(repository, "LOCAL", local)

    assert repository.relative_local_path(".local/artifacts/model") == local / "artifacts/model"
    with pytest.raises(ValueError, match="must stay under .local"):
        repository.relative_local_path("outside")


def test_atomic_write_json_uses_private_permissions(tmp_path: Path) -> None:
    destination = tmp_path / "state.json"
    repository.atomic_write_json(destination, {"status": "ok"})

    assert json.loads(destination.read_text()) == {"status": "ok"}
    assert destination.stat().st_mode & 0o777 == 0o600


def test_installed_environment_fingerprint_detects_same_version_content_change(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package.py"
    package.write_text("value = 1\n")

    class Distribution:
        metadata = {"Name": "example-package"}
        version = "1.0.0"
        files = (Path("package.py"),)

        def locate_file(self, relative: Path) -> Path:
            return tmp_path / relative

    first = repository.installed_environment_fingerprint([Distribution()])
    package.write_text("value = 2\n")
    second = repository.installed_environment_fingerprint([Distribution()])

    assert first != second


def test_executorch_checkout_rejects_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = tmp_path / "executorch"
    (checkout / ".git").mkdir(parents=True)
    responses = iter(
        (
            SimpleNamespace(stdout="expected\n"),
            SimpleNamespace(stdout="modified.py\n"),
        )
    )
    monkeypatch.setattr(repository.subprocess, "run", lambda *args, **kwargs: next(responses))

    with pytest.raises(RuntimeError, match="must remain clean"):
        repository.validate_executorch_checkout(checkout, "expected")


def test_executorch_checkout_rejects_changed_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "executorch"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(
        repository.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="different\n"),
    )

    with pytest.raises(RuntimeError, match="expected expected"):
        repository.validate_executorch_checkout(checkout, "expected")


def test_landed_gate_commits_returns_only_landed_capabilities() -> None:
    compatibility = {
        "executorch": {
            "gates": {
                "runtime": {"status": "landed", "commit": "a" * 40},
                "cancellation": {"status": "pending", "commit": None},
            }
        }
    }

    assert repository.landed_gate_commits(compatibility) == ("a" * 40,)


def test_executorch_checkout_rejects_missing_landed_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "executorch"
    (checkout / ".git").mkdir(parents=True)
    responses = iter(
        (
            SimpleNamespace(stdout="expected\n"),
            SimpleNamespace(stdout=""),
            SimpleNamespace(returncode=1),
        )
    )
    monkeypatch.setattr(repository.subprocess, "run", lambda *args, **kwargs: next(responses))

    with pytest.raises(RuntimeError, match="does not contain landed capability"):
        repository.validate_executorch_checkout(
            checkout,
            "expected",
            required_ancestors=("a" * 40,),
        )
