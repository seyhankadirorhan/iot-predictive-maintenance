"""Predict machine failure risk from a single set of sensor readings.

Run from the project root to see two example scenarios:
    python src/predict.py
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# Basic ANSI colors for readable terminal output.
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


@lru_cache(maxsize=1)
def _load_artifacts():
    model_path = MODELS_DIR / "best_model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    missing = [str(path) for path in (model_path, scaler_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Model/scaler dosyası bulunamadı: " + ", ".join(missing)
            + "\nÖnce python src/preprocess.py ve python src/train.py çalıştırın."
        )
    return joblib.load(model_path), joblib.load(scaler_path)


def _make_feature_row(
    Type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    scaler,
) -> pd.DataFrame:
    machine_type = str(Type).strip().upper()
    if machine_type not in {"L", "M", "H"}:
        raise ValueError("Type değeri 'L', 'M' veya 'H' olmalı.")

    values = {
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }
    if not all(np.isfinite(float(value)) for value in values.values()):
        raise ValueError("Tüm sensör ölçümleri sonlu sayısal değerler olmalı.")
    if float(torque) <= 0:
        raise ValueError("Torque [Nm] sıfırdan büyük olmalı.")

    row = pd.DataFrame([{"Type": machine_type, **{key: float(value) for key, value in values.items()}}])
    row["temp_diff"] = row["Process temperature [K]"] - row["Air temperature [K]"]
    row["power"] = row["Rotational speed [rpm]"] * row["Torque [Nm]"]
    row["wear_rate"] = row["Tool wear [min]"] / (row["Torque [Nm]"] + 1e-5)

    # Same pandas one-hot encoding as preprocess.py, then align to the scaler's training columns.
    row = pd.get_dummies(row, columns=["Type"], dtype=float)
    expected_features = getattr(scaler, "feature_names_in_", None)
    if expected_features is not None:
        feature_names = [str(name) for name in expected_features]
        row = row.reindex(columns=feature_names, fill_value=0)
    else:
        # Compatibility with scalers fitted on NumPy arrays without saved column names.
        row = row.reindex(columns=[
            "Type_H", "Type_L", "Type_M",
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
            "temp_diff", "power", "wear_rate",
        ], fill_value=0)
    return row


def predict_machine_status(
    Type: str,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
) -> dict[str, str | float | int]:
    """Return prediction, failure probability percentage, and warning level.

    Arguments correspond to Type, Air temperature [K], Process temperature [K],
    Rotational speed [rpm], Torque [Nm], and Tool wear [min], respectively.
    """
    model, scaler = _load_artifacts()
    feature_row = _make_feature_row(
        Type,
        air_temperature,
        process_temperature,
        rotational_speed,
        torque,
        tool_wear,
        scaler,
    )
    scaled_row = scaler.transform(feature_row)
    failure_probability = float(model.predict_proba(scaled_row)[0, 1])
    predicted_class = int(failure_probability >= 0.5)
    probability_percent = failure_probability * 100

    if probability_percent < 30:
        warning_level = "Düşük"
    elif probability_percent <= 70:
        warning_level = "Orta"
    else:
        warning_level = "Kritik"

    return {
        "prediction": "Arıza Riski Var (Failure)" if predicted_class else "Sağlam (Normal)",
        "failure_probability_percent": probability_percent,
        "warning_level": warning_level,
        "predicted_class": predicted_class,
    }


def _print_result(title: str, readings: dict[str, float | str]) -> None:
    result = predict_machine_status(**readings)
    level = str(result["warning_level"])
    color = _GREEN if level == "Düşük" else _YELLOW if level == "Orta" else _RED
    print(f"\n{_BOLD}{'─' * 64}\n{title}\n{'─' * 64}{_RESET}")
    print(
        f"Ölçümler: Type={readings['Type']}, "
        f"Air={readings['air_temperature']} K, Process={readings['process_temperature']} K, "
        f"Speed={readings['rotational_speed']} rpm, Torque={readings['torque']} Nm, "
        f"Wear={readings['tool_wear']} min"
    )
    print(f"Tahmin: {result['prediction']}")
    print(f"Arıza Olasılığı: {float(result['failure_probability_percent']):.2f}%")
    print(f"Uyarı Seviyesi: {color}{_BOLD}{level}{_RESET}")


def main() -> None:
    scenarios = [
        (
            "Örnek 1 — Normal çalışan makine",
            {
                "Type": "L",
                "air_temperature": 300.0,
                "process_temperature": 310.0,
                "rotational_speed": 1500.0,
                "torque": 35.0,
                "tool_wear": 20.0,
            },
        ),
        (
            "Örnek 2 — Yüksek sıcaklık ve aşınma",
            {
                "Type": "H",
                "air_temperature": 305.0,
                "process_temperature": 320.0,
                "rotational_speed": 1200.0,
                "torque": 58.0,
                "tool_wear": 240.0,
            },
        ),
    ]
    print(f"{_BOLD}AI4I Makine Arıza Riski Tahmini{_RESET}")
    for title, readings in scenarios:
        _print_result(title, readings)
    print()


if __name__ == "__main__":
    main()
