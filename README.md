# IoT Predictive Maintenance — AI4I 2020

IoT sensör ölçümlerini kullanarak makine arızası riskini tahmin eden uçtan uca bir kestirimci bakım (predictive maintenance) çalışması. Proje; keşifsel veri analizini, sızıntısız ön işlemeyi, özellik mühendisliğini, sınıf dengesizliğine duyarlı model eğitimini ve yeni sensör ölçümleri için tahmin üretimini kapsar.

## Veri Seti

Bu projede [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) kullanılır. Veri seti, endüstriyel kestirimci bakım senaryolarını temsil etmek üzere oluşturulmuş sentetik veridir; 10.000 gözlem ve 14 sütun içerir. Sensör ölçümlerinin yanında `Machine failure` hedefi ve `TWF`, `HDF`, `PWF`, `OSF`, `RNF` arıza türü göstergeleri bulunur. UCI açıklamasına göre veri setinde eksik değer yoktur.

Betikler `data/raw/` içinde CSV arar. Bulamazsa `eda.py` ve `preprocess.py` resmi UCI arşivini indirip CSV'yi bu klasöre kaydeder. Veri setinin Kaggle üzerinde de kopyaları bulunabilir; bu proje varsayılan olarak UCI kaynağını kullanır.

## Metodoloji ve Özellik Mühendisliği

- `UDI` ve `Product ID` tanımlayıcı sütunları özelliklerden çıkarılır.
- Hedef değişken `Machine failure` olarak ayrılır.
- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` göstergeleri özellik matrisinden çıkarılır. Bu sütunlar hedef arıza ile doğrudan ilişkili olduğundan kullanımları veri sızıntısına yol açabilir.
- Sensör ölçümlerinden aşağıdaki özellikler türetilir:

  | Özellik | Hesaplama |
  |---|---|
  | `temp_diff` | `Process temperature [K] - Air temperature [K]` |
  | `power` | `Rotational speed [rpm] * Torque [Nm]` |
  | `wear_rate` | `Tool wear [min] / (Torque [Nm] + 1e-5)` |

- `Type` varsayılan olarak One-Hot Encoding ile kodlanır. İsteğe bağlı `--encoding ordinal` seçeneği `L=1`, `M=2`, `H=3` eşlemesini kullanır.
- Veri, etiket dağılımını koruyarak (`stratify=y`) %80 eğitim ve %20 test olarak ayrılır (`random_state=42`). `StandardScaler` yalnızca eğitim bölümüne fit edilir; test bölümü aynı scaler ile dönüştürülür.
- Sınıf dengesizliğini ele almak için Random Forest `class_weight="balanced"` kullanır. XGBoost için `scale_pos_weight = negatif örnek sayısı / pozitif örnek sayısı` eğitim verisinden hesaplanır.

## Model Karşılaştırması ve Sonuçlar

Aşağıdaki metrikler projede kayıtlı modellerin ayrılmış 2.000 satırlık test kümesindeki değerlendirmesinden alınmıştır. Precision, Recall ve F1 sütunları arızalı (`Machine failure=1`) sınıfına aittir. Sınıflandırma eşiği varsayılan 0.5'tir.

| Model | ROC-AUC | Arıza Precision | Arıza Recall | Arıza F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Random Forest (`class_weight="balanced"`) | 0.9752 | 0.9016 | 0.8088 | 0.8527 | 0.9905 |
| XGBoost (`scale_pos_weight`) | **0.9836** | 0.7703 | 0.8382 | 0.8028 | 0.9860 |

`train.py`, `best_model.joblib` için ROC-AUC değeri daha yüksek modeli seçer; mevcut sonuçlarda bu model XGBoost'tur. Random Forest'ın arıza sınıfı F1 skoru daha yüksektir. Bu nedenle model seçimi, ROC-AUC ve arıza yakalama/yanlış alarm maliyetleri birlikte düşünülerek yapılmalıdır. Veri sentetik olduğundan bu sonuçlar gerçek saha performansını garanti etmez.

## Kurulum ve Hızlı Başlangıç

Python 3.10 veya üzeri önerilir. Proje kök dizininde PowerShell veya terminal açın:

```bash
pip install -r requirements.txt
```

Pipeline adımları:

```bash
# 1. Veri özeti ve EDA grafikleri
python src/eda.py

# 2. Özellik mühendisliği, stratified train/test ayırma ve ölçekleme
python src/preprocess.py

# 3. Random Forest ve XGBoost modellerini eğitip karşılaştırma
python src/train.py

# 4. Test metriklerini ve değerlendirme grafiklerini üretme
python src/evaluate.py

# 5. İki örnek sensör senaryosuyla arıza tahmini
python src/predict.py

# Web arayüzünü başlat
streamlit run app.py
```

Önceden indirilmiş bir CSV kullanmak için:

```bash
python src/eda.py --data data/raw/ai4i2020.csv
python src/preprocess.py --data data/raw/ai4i2020.csv
```

Ordinal `Type` kodlaması kullanılacaksa:

```bash
python src/preprocess.py --encoding ordinal
```

## Çıktılar

- `outputs/eda/machine_failure_distribution.png`: sağlam ve arızalı kayıt sayıları.
- `outputs/eda/failure_type_distribution.png`: TWF/HDF/PWF/OSF/RNF göstergelerinin sayıları. Bir kayıtta birden fazla tür görülebileceği için sayımlar birbirini dışlamaz.
- `outputs/eda/sensor_distributions.png`: sensör değişkenlerinin dağılımları.
- `outputs/evaluation/confusion_matrix.png`: iki modelin test kümesindeki karışıklık matrisleri.
- `outputs/evaluation/roc_curves.png`: iki modelin ROC eğrilerinin karşılaştırması.
- `outputs/evaluation/feature_importance.png`: ROC-AUC ile seçilmiş en iyi modelin özellik önemleri.
- `data/processed/train.npz`, `data/processed/test.npz`: ölçeklenmiş özellikler (`X`), hedef (`y`) ve özellik adları (`feature_names`).
- `models/scaler.joblib`: eğitim verisine fit edilmiş scaler.
- `models/rf_model.joblib`, `models/xgb_model.joblib`, `models/best_model.joblib`: eğitilmiş modeller.

## Proje Yapısı

```text
ware-house/
├── data/
│   ├── raw/                 # Kaynak CSV
│   └── processed/           # train.npz ve test.npz
├── models/                  # Scaler ve eğitilmiş modeller
├── outputs/
│   ├── eda/                 # Keşif grafikleri
│   └── evaluation/          # Model değerlendirme grafikleri
├── src/
│   ├── eda.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
└── requirements.txt
```
