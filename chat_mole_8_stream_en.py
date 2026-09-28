"""Backwards-compatible shim. Real implementation: biglittle_moe.chat"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from biglittle_moe.chat import main

if __name__ == "__main__":
    main()
