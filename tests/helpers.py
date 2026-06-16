from __future__ import annotations

from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_TMP_ROOT = PROJECT_ROOT / ".test_tmp"


def make_test_dir() -> Path:
    path = TEST_TMP_ROOT / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path

