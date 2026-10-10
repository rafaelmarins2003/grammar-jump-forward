#!/usr/bin/env python3
"""Run the bounded diagnosis and replication stage."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jumpforward.replication import main  # noqa: E402

main()
