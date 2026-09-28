"""CLI entry point for corpus assembly."""

from __future__ import annotations

from pathlib import Path

from .config import DATA_PATH
from .data import build_corpus


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Download and assemble MoLE training data")
    ap.add_argument("--out", type=Path, default=DATA_PATH)
    args = ap.parse_args()

    n = build_corpus(args.out)
    print(f"[+] Done: {n:,} samples -> {args.out}")


if __name__ == "__main__":
    main()
