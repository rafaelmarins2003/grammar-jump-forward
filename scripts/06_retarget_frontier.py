"""Build the retargeting report from recorded prompts, cache probes and reference traces."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jumpforward.frontier import main  # noqa: E402

if __name__ == "__main__":
    main()
