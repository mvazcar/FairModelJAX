"""CLI entry point — thin wrapper so ``python run.py ...`` works without install."""
import sys
from pathlib import Path

# Allow running from the repo without installing.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pyfair.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
