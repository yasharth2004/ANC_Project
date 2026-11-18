#!/usr/bin/env python3
"""Validate that the HybridANC Python dependencies can be imported."""

from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = {
    "numpy": "NumPy",
    "scipy": "SciPy",
    "sounddevice": "SoundDevice",
    "matplotlib": "Matplotlib",
    "pytest": "pytest",
    "pytest_cov": "pytest-cov",
    "soundfile": "SoundFile",
}


def main() -> int:
    failed: list[str] = []

    for module_name, display_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            print(f"FAIL: {display_name}: {exc}", file=sys.stderr)
            failed.append(display_name)
            continue

        version = getattr(module, "__version__", "unknown")
        print(f"OK:   {display_name} {version}")

    if failed:
        print(
            "Missing or broken dependencies: " + ", ".join(failed),
            file=sys.stderr,
        )
        return 1

    print("All HybridANC dependencies are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
