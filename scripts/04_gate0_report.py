"""Render the Gate 0 decision from the completed measurement artifacts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jumpforward.report import main  # noqa: E402

if __name__ == "__main__":
    main()
