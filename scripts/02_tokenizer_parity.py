"""Run the local Qwen3 GGUF tokenizer parity check without downloading models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jumpforward.parity import main  # noqa: E402

if __name__ == "__main__":
    main()
