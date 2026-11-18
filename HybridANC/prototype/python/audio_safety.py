"""Prototype-safe audio-safety demonstration and package entry point."""

from __future__ import annotations

import argparse
import logging

import numpy as np

try:
    from ...core.safety.audio_safety import (
        AudioSafetyConfig,
        AudioSafetyController,
        SafetyState,
    )
except ImportError:
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from core.safety.audio_safety import (
        AudioSafetyConfig,
        AudioSafetyController,
        SafetyState,
    )


LOGGER = logging.getLogger("hybridanc.audio_safety")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demonstrate HybridANC output safety controls"
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=0.05,
        help="Startup gain, capped by conservative safety limits",
    )
    parser.add_argument(
        "--emergency-test",
        action="store_true",
        help="Trigger emergency mute after startup",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    safety = AudioSafetyController(
        AudioSafetyConfig(
            maximum_gain=0.25,
            startup_gain=min(args.gain, 0.05),
        )
    )
    if not safety.start_safe():
        return 1

    if args.emergency_test:
        safety.emergency_mute("demonstration")
    print(f"Safety state: {safety.state.value}")
    print(f"Muted: {safety.muted}")
    print(f"Physical ANC enabled: {safety.physical_anc.enabled}")
    print(
        "Soft clipping: "
        f"{np.max(np.abs(safety.soft_clip(np.array([2.0, -2.0])))):.6f}"
    )

    safety.safe_shutdown()
    print(f"Safety state after shutdown: {safety.state.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
