"""Train and compare class-imbalance-aware failure classifiers.

Run from the project root after preprocessing:
    python src/train.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


def load_split(path: Path):
    """Load X, y, and feature names from a preprocessing NPZ file."""
    if not path.is_file():
        raise FileNotFoundError(f"İşlenmiş veri dosyası bulunamadı: {path}\nÖnce python src/preprocess.py çalıştırın.")
    with np.load(path, allow_pickle=False) as data:
        required = {"X", "y", "feature_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path.name} içinde beklenen alanlar eksik: {sorted(missing)}")
        X = data["X"]
        y = data["y"].ravel()
        feature_names = data["feature_names"].astype(str).tolist()
    return X, y, feature_names


def main() -> None:
    X_train, y_train, train_features = load_split(PROCESSED_DIR / "train.npz")
    X_test, y_test, test_features = load_split(PROCESSED_DIR / "test.npz")

    if train_features != test_features:
        raise ValueError("Train ve test özellik adları/sıraları uyuşmuyor.")
    if X_train.shape[1] != X_test.shape[1] or X_train.shape[1] != len(train_features):
        raise ValueError("Train/test özellik boyutları ya da özellik adları uyuşmuyor.")
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Eğitim hedefi 0 ve 1 sınıflarını içermeli.")
    if set(np.unique(y_test)) != {0, 1}:
        raise ValueError("Test hedefi 0 ve 1 sınıflarını içermeli.")

    negative_count = int(np.count_nonzero(y_train == 0))
    positive_count = int(np.count_nonzero(y_train == 1))
    if positive_count == 0:
        raise ValueError("Eğitim verisinde pozitif (Machine failure=1) örnek yok.")
    scale_pos_weight = negative_count / positive_count
    print(f"Train sınıf sayıları: sağlam={negative_count}, arızalı={positive_count}")
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}")

    rf_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    print("Random Forest eğitiliyor...")
    rf_model.fit(X_train, y_train)
    print("XGBoost eğitiliyor...")
    xgb_model.fit(X_train, y_train)

    results = {}
    for name, model in (("Random Forest", rf_model), ("XGBoost", xgb_model)):
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probabilities)
        f1 = f1_score(y_test, predictions, zero_division=0)
        results[name] = {"roc_auc": auc, "f1": f1}
        print(f"{name}: ROC-AUC={auc:.4f} | F1={f1:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rf_path = MODELS_DIR / "rf_model.joblib"
    xgb_path = MODELS_DIR / "xgb_model.joblib"
    best_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_path)

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = rf_model if best_name == "Random Forest" else xgb_model
    joblib.dump(best_model, best_path)
    print(f"ROC-AUC ile seçilen en iyi model: {best_name} ({results[best_name]['roc_auc']:.4f})")
    print(f"Kaydedildi: {rf_path}")
    print(f"Kaydedildi: {xgb_path}")
    print(f"Kaydedildi: {best_path}")


if __name__ == "__main__":
    main()
