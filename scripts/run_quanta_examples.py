"""Run Python, optimized C, or both on Quanta example ECG files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python_model.runtime import MODEL_NAMES, TINYAF_MODELS
from src.run_example import predict_c, predict_python

SAMPLE_RATE_HZ = 256
SAMPLE_COUNT = 2560
ADC_MIN = 0
ADC_MAX = 4095


def load_quanta_example(path: Path) -> dict:
    adc = np.loadtxt(path, dtype=np.int64)

    if adc.ndim != 1:
        raise ValueError(f"{path.name}: expected a 1D ECG, got shape {adc.shape}")
    if adc.size != SAMPLE_COUNT:
        raise ValueError(
            f"{path.name}: expected {SAMPLE_COUNT} samples, got {adc.size}"
        )
    if adc.min() < ADC_MIN or adc.max() > ADC_MAX:
        raise ValueError(
            f"{path.name}: expected 12-bit unsigned values in "
            f"{ADC_MIN}..{ADC_MAX}, got {adc.min()}..{adc.max()}"
        )

    adc = adc.astype(np.uint16)
    return {
        "name": path.stem,
        "path": path,
        "adc": adc,
    }


def load_examples(examples_dir: Path) -> list[dict]:
    example_paths = sorted(examples_dir.glob("*.txt"))
    if not example_paths:
        raise FileNotFoundError(
            f"No Quanta ECG example .txt files found in {examples_dir}"
        )

    return [load_quanta_example(path) for path in example_paths]


def get_names_and_ecg_array(
    examples: list[dict], dtype: np.dtype
) -> tuple[list[str], np.ndarray]:
    names = [example["name"] for example in examples]
    ecg_array = np.stack([example["adc"].astype(dtype) for example in examples]).astype(
        dtype
    )
    return names, ecg_array


def prediction_rows(
    implementation: str,
    model_name: str,
    names: list[str],
    probabilities: np.ndarray,
) -> list[dict]:
    probabilities = np.asarray(probabilities).reshape(-1)

    if len(probabilities) != len(names):
        raise RuntimeError(
            f"Expected {len(names)} probabilities, got {probabilities.shape[0]}"
        )

    rows = []
    for name, probability in zip(names, probabilities):
        rows.append(
            {
                "implementation": implementation,
                "example": name,
                "model": model_name,
                "probability": float(probability),
            }
        )

    return rows


def predict_python_model(model_name: str, examples: list[dict], device="cpu") -> list[dict]:
    names, ecg_array = get_names_and_ecg_array(examples, np.uint16)
    probabilities = predict_python(
        ecg_array,
        model=model_name,
        sample_rate=SAMPLE_RATE_HZ,
        quanta_adc_to_physical=True,
        device=device,
    )
    return prediction_rows("python", model_name, names, probabilities)


def predict_c_model(model_name: str, examples: list[dict]) -> list[dict]:
    names, ecg_array = get_names_and_ecg_array(examples, np.uint16)
    probabilities = predict_c(
        ecg_array,
        model=model_name,
    )
    return prediction_rows("c", model_name, names, probabilities)


def print_predictions(rows: list[dict]) -> None:
    print("implementation,example,model,probability")
    for row in rows:
        print(
            f"{row['implementation']},{row['example']},{row['model']},{row['probability']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Python, optimized C, or both on 12-bit Quanta ECG example files. "
            "Run from the tinyaf repo root so model_assets/ and c_model/ resolve."
        )
    )
    parser.add_argument(
        "--examples-dir",
        type=Path,
        required=True,
        help="Directory containing one-value-per-line Quanta ECG .txt files.",
    )
    parser.add_argument(
        "--implementation",
        choices=("python", "c", "both"),
        default="both",
        help=(
            "Implementation to run: python needs no C build tools; c runs optimized C; "
            "both compares Python and optimized C (default: both)."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str.upper,
        choices=MODEL_NAMES,
        default=["TINYAF-S"],
        help="TINYAF-S, TINYAF-M, TINYAF-L, LARGENN, SMALLNN, or HRHRV.",
    )
    parser.add_argument("--nn-device", default="cpu", help="Device for all Python models, e.g. cuda:0.")
    args = parser.parse_args()
    if args.implementation == "c" and any(m not in TINYAF_MODELS for m in args.models):
        parser.error("Baseline models support Python only")
    return args


def main() -> int:
    args = parse_args()

    examples = load_examples(args.examples_dir)

    output_rows = []
    if args.implementation in ("python", "both"):
        for model_name in args.models:
            output_rows.extend(predict_python_model(model_name, examples, args.nn_device))

    if args.implementation in ("c", "both"):
        for model_name in args.models:
            if model_name not in TINYAF_MODELS:
                continue
            output_rows.extend(predict_c_model(model_name, examples))

    print_predictions(output_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
