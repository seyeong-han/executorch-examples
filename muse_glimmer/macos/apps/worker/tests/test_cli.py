from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from muse_glimmer_worker import cli


def test_console_command_uses_installed_module() -> None:
    args = argparse.Namespace(
        input_device="Built-in Mic",
        output_device="Built-in Output",
        list_devices=True,
        text=True,
        record=False,
        console_log_level="info",
    )
    assert cli._console_command(args) == [
        sys.executable,
        "-m",
        "muse_glimmer_worker",
        "console",
        "--log-level",
        "info",
        "--input-device",
        "Built-in Mic",
        "--output-device",
        "Built-in Output",
        "--list-devices",
        "--text",
    ]


def test_issue_bundle_omits_runtime_log(tmp_path: Path) -> None:
    for name in ("report.json", "events.jsonl", "runtime.log"):
        (tmp_path / name).write_text(name)

    bundle = cli._create_issue_bundle(tmp_path)

    with zipfile.ZipFile(bundle) as archive:
        assert set(archive.namelist()) == {"report.json", "events.jsonl"}
