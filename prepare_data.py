"""Backwards-compatible shim. Real implementation: biglittle_moe.prepare"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from biglittle_moe.prepare import main

if __name__ == "__main__":
    main()
