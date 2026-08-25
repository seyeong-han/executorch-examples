from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import privacy_audit


def test_external_connection_audit_includes_managed_process_group(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / ".local/run"
    run_dir.mkdir(parents=True)
    (run_dir / "stack.json").write_text(
        json.dumps({"services": {"web": {"pid": 100, "pgid": 100}}})
    )
    output = (
        "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
        "node 321 user 10u IPv4 1 0t0 TCP 127.0.0.1:5000->203.0.113.10:443\n"
    )
    monkeypatch.setattr(privacy_audit, "ROOT", tmp_path)
    monkeypatch.setattr(
        privacy_audit.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )
    monkeypatch.setattr(privacy_audit.os, "getpgid", lambda pid: 100 if pid == 321 else pid)

    with pytest.raises(RuntimeError, match="non-loopback connection"):
        privacy_audit._assert_no_external_connections()
