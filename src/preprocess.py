"""AI4I 2020 preprocessing pipeline. Run: python src/preprocess.py"""
from __future__ import annotations
import argparse
import io
import os
import urllib.request
import zipfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
TARGET = "Machine failure"
ID_COLUMNS = ["UDI", "Product ID"]
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
AIR_TEMP = "Air temperature [K]"
PROCESS_TEMP = "Process temperature [K]"
ROT_SPEED = "Rotational speed [rpm]"
TORQUE = "Torque [Nm]"
TOOL_WEAR = "Tool wear [min]"
SENSOR_COLUMNS = [AIR_TEMP, PROCESS_TEMP, ROT_SPEED, TORQUE, TOOL_WEAR]


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    """Read a supplied/local CSV or download the official UCI archive."""
    if csv_path is not None:
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")
        return pd.read_csv(csv_path)
    raw_dir = PROJECT_ROOT / "data" / "raw"
    candidates = [
        raw_dir / "ai4i2020.csv",
        raw_dir / "ai4i2020+predictive+maintenance+dataset.csv",
        raw_dir / "ai4i2020_predictive_maintenance_dataset.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            print(f"Yerel veri dosyası kullanılıyor: {candidate}")
            return pd.read_csv(candidate)
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("AI4I veri seti UCI deposundan indiriliyor...")
    request = urllib.request.Request(UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; AI4I-Preprocess/1.0)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise FileNotFoundError("UCI arşivinde CSV dosyası bulunamadı.")
        member = next((name for name in members if "ai4i2020" in name.lower()), members[0])
        destination = raw_dir / "ai4i2020.csv"
        destination.write_bytes(archive.read(member))
    print(f"Veri kaydedildi: {destination}")
    return pd.read_csv(destination)


def prepare_data(df: pd.DataFrame, encoding: str = "onehot", random_state: int = 42):
    """Engineer features, split stratified, and fit scaling on train only."""
    required = [TARGET, "Type", *SENSOR_COLUMNS, *ID_COLUMNS, *LEAKAGE_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("Gerekli sütun(lar) eksik: " + ", ".join(missing))

    # Separate target, IDs, and failure-mode labels before constructing X.
    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET, *ID_COLUMNS, *LEAKAGE_COLUMNS]).copy()
    X["temp_diff"] = X[PROCESS_TEMP] - X[AIR_TEMP]
    X["power"] = X[ROT_SPEED] * X[TORQUE]
    X["wear_rate"] = X[TOOL_WEAR] / (X[TORQUE] + 1e-5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    if encoding == "onehot":
        X_train = pd.get_dummies(X_train, columns=["Type"], dtype=float)
        X_test = pd.get_dummies(X_test, columns=["Type"], dtype=float)
        # Any unseen test category is represented as all zeros.
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    elif encoding == "ordinal":
        type_map = {"L": 1, "M": 2, "H": 3}
        for split_name, split in (("train", X_train), ("test", X_test)):
            mapped = split["Type"].map(type_map)
            if mapped.isna().any():
                unknown = sorted(split.loc[mapped.isna(), "Type"].astype(str).unique())
                raise ValueError(f"Tanınmayan Type değeri: {unknown} ({split_name})")
            split["Type"] = mapped.astype(int)
    else:
        raise ValueError("encoding 'onehot' veya 'ordinal' olmalıdır.")

    feature_names = X_train.columns.tolist()
    if any(name in feature_names for name in LEAKAGE_COLUMNS):
        raise RuntimeError("Veri sızıntısı kontrolü başarısız: arıza tipi X içinde kaldı.")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train.to_numpy(), y_test.to_numpy(), feature_names, scaler


def save_outputs(X_train, X_test, y_train, y_test, feature_names, scaler, processed_dir: Path, models_dir: Path):
    processed_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_dir / "scaler.joblib"
    os.makedirs(scaler_path.parent, exist_ok=True)
    np.savez_compressed(processed_dir / "train.npz", X=X_train, y=y_train, feature_names=np.asarray(feature_names, dtype=str))
    np.savez_compressed(processed_dir / "test.npz", X=X_test, y=y_test, feature_names=np.asarray(feature_names, dtype=str))
    joblib.dump(scaler, scaler_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI4I 2020 veri ön işleme pipeline'ı")
    parser.add_argument("--data", type=Path, help="CSV yolu; verilmezse data/raw aranır, sonra UCI'den indirilir.")
    parser.add_argument("--encoding", choices=["onehot", "ordinal"], default="onehot", help="Type kodlaması.")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()
    csv_path = args.data.expanduser().resolve() if args.data else None
    processed_dir, models_dir = args.processed_dir.expanduser(), args.models_dir.expanduser()
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir
    if not models_dir.is_absolute():
        models_dir = PROJECT_ROOT / models_dir

    df = load_dataset(csv_path)
    Xtr, Xte, ytr, yte, names, scaler = prepare_data(df, args.encoding)
    save_outputs(Xtr, Xte, ytr, yte, names, scaler, processed_dir.resolve(), models_dir.resolve())
    print(f"Train: X={Xtr.shape}, y={ytr.shape}")
    print(f"Test: X={Xte.shape}, y={yte.shape}")
    print(f"Özellikler: {', '.join(names)}")
    print(f"Train/test dosyaları: {processed_dir.resolve()}")
    print(f"Scaler: {(models_dir / 'scaler.joblib').resolve()}")


if __name__ == "__main__":
    main()
