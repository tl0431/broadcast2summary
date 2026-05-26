# tests/conftest.py
import logging
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _reset_broadcast2summary_logger():
    """configure_run_logging() sets propagate=False and adds a FileHandler on the
    broadcast2summary logger — these are module-level mutations that leak across
    tests and break pytest's caplog (which captures via root). Reset before each test.
    """
    log = logging.getLogger("broadcast2summary")
    log.propagate = True
    for h in list(log.handlers):
        log.removeHandler(h)
    yield


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
