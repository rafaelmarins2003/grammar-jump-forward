"""Generate the preregistered small whitespace comparison without engine changes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jumpforward.generation import main  # noqa: E402

if __name__ == "__main__":
    main()
