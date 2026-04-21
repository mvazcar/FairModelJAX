"""Make ``src/`` importable for pytest without requiring ``pip install -e .``."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "src"))


def pytest_addoption(parser):
    """Add ``--runslow`` flag for tests marked ``@pytest.mark.slow``."""
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="Run slow tests (e.g. full FP.EXE regeneration)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
