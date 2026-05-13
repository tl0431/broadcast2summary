# tests/conftest.py
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def tmp_state_dir(tmp_path) -> Path:
    d = tmp_path / "state"
    d.mkdir()
    (d / "audio").mkdir()
    (d / "failed").mkdir()
    return d
