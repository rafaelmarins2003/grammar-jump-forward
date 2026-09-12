"""Freeze, measure or summarize existing general tool-calling baselines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jumpforward.baselines import main  # noqa: E402

if __name__ == "__main__":
    main()
