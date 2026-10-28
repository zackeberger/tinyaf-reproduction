#!/usr/bin/env python3
"""Resample 10-second Quanta ECG text files from 100 Hz to 256 Hz."""

from __future__ import annotations

import argparse
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


SOURCE_RATE_HZ = 100
TARGET_RATE_HZ = 256
DURATION_SECONDS = 10
SOURCE_SAMPLE_COUNT = SOURCE_RATE_HZ * DURATION_SECONDS
TARGET_SAMPLE_COUNT = TARGET_RATE_HZ * DURATION_SECONDS
ADC_MIN = 0
ADC_MAX = 4095


def load_quanta_ecg(path: Path) -> np.ndarray:
    """Load and validate one 10-second, 100 Hz Quanta ECG."""
    values = np.loadtxt(path, dtype=np.int64)

    if values.ndim != 1:
        raise ValueError(f"{path}: expected one value per line, got shape {values.shape}")
    if values.size != SOURCE_SAMPLE_COUNT:
        raise ValueError(
            f"{path}: expected {SOURCE_SAMPLE_COUNT} samples at {SOURCE_RATE_HZ} Hz, "
            f"got {values.size}"
        )
    if values.min() < ADC_MIN or values.max() > ADC_MAX:
        raise ValueError(
            f"{path}: expected 12-bit unsigned ADC values in {ADC_MIN}..{ADC_MAX}, "
            f"got {values.min()}..{values.max()}"
        )

    return values


def resample_quanta_ecg(values: np.ndarray) -> np.ndarray:
    """Polyphase-resample Quanta ADC counts and return valid uint16 counts."""
    divisor = gcd(SOURCE_RATE_HZ, TARGET_RATE_HZ)
    resampled = resample_poly(
        values.astype(np.float64),
        up=TARGET_RATE_HZ // divisor,
        down=SOURCE_RATE_HZ // divisor,
        padtype="line",
    )

    if resampled.size != TARGET_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_SAMPLE_COUNT} resampled values, got {resampled.size}"
        )

    return np.clip(np.rint(resampled), ADC_MIN, ADC_MAX).astype(np.uint16)


def resample_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    """Resample every .txt file in input_dir into output_dir."""
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("input_dir and output_dir must be different directories")

    input_paths = sorted(input_dir.glob("*.txt"))
    if not input_paths:
        raise FileNotFoundError(f"No .txt ECG files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for input_path in input_paths:
        output_path = output_dir / input_path.name
        resampled = resample_quanta_ecg(load_quanta_ecg(input_path))
        np.savetxt(output_path, resampled, fmt="%d")
        output_paths.append(output_path)
        print(f"{input_path} -> {output_path} ({resampled.size} samples)")

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resample 10-second, one-value-per-line, 12-bit Quanta ECG files "
            "from 100 Hz (1,000 samples) to 256 Hz (2,560 samples)."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory of 100 Hz .txt files")
    parser.add_argument("output_dir", type=Path, help="Destination for 256 Hz .txt files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resample_directory(args.input_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
