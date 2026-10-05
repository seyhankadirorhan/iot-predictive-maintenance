"""Load the UCI AI4I 2020 dataset and produce a compact exploratory analysis.

Usage from the project root:
    python src/eda.py
    python src/eda.py --data data/raw/ai4i2020.csv --output-dir outputs/eda

If no local CSV is found, the official UCI ZIP archive is downloaded to data/raw.
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FAILURE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_dataset(csv_path: Path | None = None) -> pd.DataFrame:
    """Load a provided/local CSV, or download the official UCI archive."""
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
    request = urllib.request.Request(
        UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; AI4I-EDA/1.0)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError("UCI arşivinde CSV dosyası bulunamadı.")
        # Prefer the canonical AI4I CSV if the archive contains more than one CSV.
        member = next((name for name in csv_members if "ai4i2020" in name.lower()), csv_members[0])
        destination = raw_dir / "ai4i2020.csv"
        destination.write_bytes(archive.read(member))
    print(f"Veri kaydedildi: {destination}")
    return pd.read_csv(destination)


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Veri seti boyutu ===")
    print(f"Satır: {len(df):,} | Sütun: {len(df.columns)}")

    print("\n=== Sütun tipleri ===")
    print(df.dtypes.to_string())

    print("\n=== Eksik değerler ===")
    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = (missing["missing_count"] / max(len(df), 1) * 100).round(2)
    print(missing.to_string())
    print(f"Toplam eksik hücre: {int(df.isna().sum().sum())}")

    available_sensors = [column for column in SENSOR_COLUMNS if column in df.columns]
    absent_sensors = [column for column in SENSOR_COLUMNS if column not in df.columns]
    print("\n=== Sensör özet istatistikleri ===")
    if available_sensors:
        print(df[available_sensors].describe().T.to_string())
    else:
        print("Beklenen sensör sütunları bulunamadı.")
    if absent_sensors:
        print("Eksik sensör sütunları: " + ", ".join(absent_sensors))


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    if "Machine failure" in df.columns:
        counts = df["Machine failure"].value_counts().reindex([0, 1], fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(counts.index.astype(str), counts.values,
               color=["#4C78A8", "#E45756"])
        ax.set(title="Machine failure sınıf dağılımı", xlabel="Machine failure (0 = hayır, 1 = evet)", ylabel="Kayıt sayısı")
        for bar, value in zip(ax.patches, counts.values):
            ax.annotate(f"{value:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
        fig.tight_layout()
        fig.savefig(output_dir / "machine_failure_distribution.png", dpi=160)
        plt.close(fig)
    else:
        print("Uyarı: 'Machine failure' sütunu yok; sınıf grafiği atlandı.")

    available_failures = [column for column in FAILURE_COLUMNS if column in df.columns]
    if available_failures:
        # Failure indicators can overlap; these are per-mode positive counts.
        failure_counts = df[available_failures].fillna(0).astype(int).sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = sns.color_palette("mako", n_colors=len(failure_counts))
        ax.bar(failure_counts.index, failure_counts.values, color=colors)
        ax.set(title="Failure tipleri (pozitif kayıt sayısı)", xlabel="Failure tipi", ylabel="Kayıt sayısı")
        ax.bar_label(ax.containers[0], padding=3)
        ax.text(0.99, 0.98, "Tipler çakışabilir; toplam Machine failure sayısına eşit olmak zorunda değildir.",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555555")
        fig.tight_layout()
        fig.savefig(output_dir / "failure_type_distribution.png", dpi=160)
        plt.close(fig)
    else:
        print("Uyarı: TWF/HDF/PWF/OSF/RNF sütunları yok; failure tipi grafiği atlandı.")

    available_sensors = [column for column in SENSOR_COLUMNS if column in df.columns]
    if available_sensors:
        fig, axes = plt.subplots(len(available_sensors), 1, figsize=(10, 2.6 * len(available_sensors)))
        if len(available_sensors) == 1:
            axes = [axes]
        for ax, column in zip(axes, available_sensors):
            sns.histplot(data=df, x=column, hue="Machine failure" if "Machine failure" in df.columns else None,
                         bins=35, stat="count", common_norm=False, element="step", ax=ax)
            ax.set_title(column)
            ax.set_ylabel("Kayıt sayısı")
        fig.suptitle("Sensör değerlerinin dağılımı", y=1.002, fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / "sensor_distributions.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"\nGrafikler kaydedildi: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI4I 2020 temel EDA betiği")
    parser.add_argument("--data", type=Path, help="Kullanılacak CSV yolu (verilmezse yerel dosya aranır, sonra UCI'den indirilir).")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "eda",
                        help="Grafiklerin kaydedileceği klasör (varsayılan: outputs/eda).")
    args = parser.parse_args()

    csv_path = args.data.expanduser().resolve() if args.data else None
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()

    df = load_dataset(csv_path)
    print_summary(df)
    make_plots(df, output_dir)


if __name__ == "__main__":
    main()
