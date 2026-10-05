"""Evaluate trained AI4I failure classifiers and save comparison plots.

Run from the project root after preprocessing and training:
    python src/evaluate.py
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.npz"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def load_test_data(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Test verisi bulunamadı: {path}\nÖnce python src/preprocess.py çalıştırın.")
    with np.load(path, allow_pickle=False) as data:
        required = {"X", "y", "feature_names"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"test.npz içinde beklenen alanlar eksik: {sorted(missing)}")
        X_test = data["X"]
        y_test = data["y"].ravel()
        feature_names = data["feature_names"].astype(str).tolist()
    return X_test, y_test, feature_names


def load_model(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {path}\nÖnce python src/train.py çalıştırın.")
    return joblib.load(path)


def main() -> None:
    # Create output directory before generating any artifacts.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_test, y_test, feature_names = load_test_data(TEST_PATH)
    models = {
        "Random Forest": load_model(MODELS_DIR / "rf_model.joblib"),
        "XGBoost": load_model(MODELS_DIR / "xgb_model.joblib"),
    }
    best_model = load_model(MODELS_DIR / "best_model.joblib")

    sns.set_theme(style="whitegrid", context="notebook")
    predictions = {}
    probabilities = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
        probabilities[name] = model.predict_proba(X_test)[:, 1]
        print(f"\n{'=' * 72}\n{name} — Classification Report\n{'=' * 72}")
        print(classification_report(
            y_test,
            predictions[name],
            labels=[0, 1],
            target_names=["Sağlam (0)", "Arızalı (1)"],
            digits=4,
            zero_division=0,
        ))
        print(f"ROC-AUC: {roc_auc_score(y_test, probabilities[name]):.4f}")

    # Confusion matrices in one side-by-side image.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, y_pred) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Sağlam (0)", "Arızalı (1)"],
            yticklabels=["Sağlam (0)", "Arızalı (1)"],
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
    fig.suptitle("Confusion Matrix — Test Seti", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ROC curve comparison.
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, y_score in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Rastgele sınıflandırıcı")
    ax.set(title="ROC Eğrileri — Test Seti", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Feature importances for the model selected during training.
    if not hasattr(best_model, "feature_importances_"):
        raise AttributeError("best_model.joblib özellik önemlerini sunmuyor (feature_importances_ yok).")
    importances = np.asarray(best_model.feature_importances_)
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Özellik sayısı uyuşmuyor: model={len(importances)}, test.npz={len(feature_names)}."
        )
    order = np.argsort(importances)[::-1]
    fig_height = max(5, 0.38 * len(feature_names))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(np.asarray(feature_names)[order][::-1], importances[order][::-1], color="#4C78A8")
    ax.set(title="Özellik Önemi — Best Model", xlabel="Importance", ylabel="Özellik")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDeğerlendirme grafikleri kaydedildi: {OUTPUT_DIR}")
    print(" - confusion_matrix.png")
    print(" - roc_curves.png")
    print(" - feature_importance.png")


if __name__ == "__main__":
    main()
