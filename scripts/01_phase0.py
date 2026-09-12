"""Run with .venv/bin/python scripts/01_phase0.py --help (works after directory moves)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jumpforward.phase0 import main  # noqa: E402

if __name__ == "__main__":
    main()
