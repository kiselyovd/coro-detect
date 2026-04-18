"""Smoke-test the data sync helper idempotence + placement."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="bash path-interop on Windows makes subprocess(['bash', ...]) unreliable; CI runs on Linux.",
)
def test_sync_data_idempotent_when_csv_already_present(tmp_path: Path, monkeypatch) -> None:
    work = tmp_path / "repo"
    (work / "data" / "raw").mkdir(parents=True)
    target = work / "data" / "raw" / "cardio.csv"
    target.write_text("dummy,header\n1,2\n", encoding="utf-8")

    script = Path(__file__).resolve().parents[1] / "scripts" / "sync_data.sh"
    assert script.exists(), "scripts/sync_data.sh missing"

    env = os.environ.copy()
    env["CARDIO_REPO_ROOT"] = str(work)
    result = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, cwd=work
    )
    assert result.returncode == 0, result.stderr
    assert target.read_text(encoding="utf-8").startswith("dummy,header")
