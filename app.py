"""Streamlit dashboard for AI4I 2020 predictive maintenance."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import predict_machine_status  # noqa: E402

st.set_page_config(
    page_title="IoT Kestirimci Bakım & Arıza Tahmin Paneli",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      .hero {padding: 1.35rem 1.6rem; border-radius: 16px;
             background: linear-gradient(120deg, #0f172a 0%, #123b5d 65%, #087e8b 100%);
             color: white; margin-bottom: 1.25rem;}
      .hero h1 {font-size: 2rem; margin: 0 0 .35rem 0; color: white;}
      .hero p {font-size: 1rem; opacity: .9; margin: 0;}
      .risk-low {border-left: 5px solid #16a34a; background: #f0fdf4;
                 border-radius: 8px; padding: .8rem 1rem; color: #14532d;}
      .risk-medium {border-left: 5px solid #eab308; background: #fefce8;
                    border-radius: 8px; padding: .8rem 1rem; color: #713f12;}
      .risk-critical {border-left: 5px solid #dc2626; background: #fef2f2;
                      border-radius: 8px; padding: .8rem 1rem; color: #7f1d1d;}
      div[data-testid="stMetric"] {background: #f8fafc; padding: .85rem 1rem;
                                    border-radius: 12px; border: 1px solid #e2e8f0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Seed the widget state before creating the corresponding widgets.
default_values = {
    "machine_type": "L",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500,
    "torque": 40.0,
    "tool_wear": 30,
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

normal_values = {
    "machine_type": "L",
    "air_temperature": 300.0,
    "process_temperature": 310.0,
    "rotational_speed": 1500,
    "torque": 35.0,
    "tool_wear": 20,
}
critical_values = {
    "machine_type": "H",
    "air_temperature": 304.5,
    "process_temperature": 314.8,
    "rotational_speed": 1200,
    "torque": 58.0,
    "tool_wear": 245,
}

with st.sidebar:
    st.header("🧭 Sensör Girdileri")
    st.caption("Makine ölçümlerini girin veya hızlı test senaryolarından birini seçin.")
    preset_col1, preset_col2 = st.columns(2)
    if preset_col1.button("Normal Çalışma Değerlerini Yükle", use_container_width=True):
        st.session_state.update(normal_values)
    if preset_col2.button("Kritik Risk Değerlerini Yükle", use_container_width=True):
        st.session_state.update(critical_values)

    st.selectbox("Makine Tipi", options=["L", "M", "H"], key="machine_type",
                 help="L: düşük, M: orta, H: yüksek ürün kalitesi varyantı.")
    st.slider("Hava Sıcaklığı [K]", min_value=295.0, max_value=305.0,
              value=300.0, step=0.1, key="air_temperature")
    st.slider("Proses Sıcaklığı [K]", min_value=305.0, max_value=315.0,
              value=310.0, step=0.1, key="process_temperature")
    st.slider("Dönme Hızı [rpm]", min_value=1100, max_value=2900,
              value=1500, step=10, key="rotational_speed")
    st.slider("Tork [Nm]", min_value=3.0, max_value=80.0,
              value=40.0, step=0.5, key="torque")
    st.slider("Takım Aşınması [min]", min_value=0, max_value=260,
              value=30, step=1, key="tool_wear")

    analyze = st.button("🔍 Makine Durumunu Analiz Et", type="primary", use_container_width=True)
    st.divider()
    st.caption("Uygulamayı başlatma komutu")
    st.code("streamlit run app.py", language="powershell")

st.markdown(
    """
    <div class="hero">
      <h1>🛠️ IoT Kestirimci Bakım &amp; Arıza Tahmin Paneli</h1>
      <p>AI4I 2020 sensör verileriyle makine sağlığını izleyin, arıza olasılığını değerlendirin
      ve bakım kararlarını destekleyin.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

current_inputs = {
    "Type": st.session_state.machine_type,
    "air_temperature": st.session_state.air_temperature,
    "process_temperature": st.session_state.process_temperature,
    "rotational_speed": st.session_state.rotational_speed,
    "torque": st.session_state.torque,
    "tool_wear": st.session_state.tool_wear,
}

if analyze:
    try:
        st.session_state.last_result = predict_machine_status(**current_inputs)
        st.session_state.last_inputs = current_inputs.copy()
    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Tahmin sırasında hata oluştu: {exc}")

if "last_result" in st.session_state:
    result = st.session_state.last_result
    probability = float(result["failure_probability_percent"])
    prediction = str(result["prediction"])
    warning = str(result["warning_level"])

    result_col, probability_col, warning_col = st.columns([1.4, 1, 1])
    with result_col:
        if int(result["predicted_class"]) == 0:
            st.success(f"### ✅ {prediction}")
        else:
            st.error(f"### ⚠️ {prediction}")
    with probability_col:
        st.metric("Arıza Olasılığı", f"{probability:.2f}%")
        st.progress(min(max(probability / 100.0, 0.0), 1.0), text="Arıza riski")
    with warning_col:
        css_class = "risk-low" if warning == "Düşük" else "risk-medium" if warning == "Orta" else "risk-critical"
        icon = "🟢" if warning == "Düşük" else "🟡" if warning == "Orta" else "🔴"
        st.markdown(
            f'<div class="{css_class}"><strong>Uyarı Seviyesi</strong><br>'
            f'<span style="font-size:1.35rem">{icon} {warning}</span></div>',
            unsafe_allow_html=True,
        )

    result_inputs = st.session_state.get("last_inputs", current_inputs)
    temp_diff = result_inputs["process_temperature"] - result_inputs["air_temperature"]
    power = result_inputs["rotational_speed"] * result_inputs["torque"]
    wear_rate = result_inputs["tool_wear"] / (result_inputs["torque"] + 1e-5)
    st.subheader("Türetilen Özellikler")
    feature_cols = st.columns(3)
    feature_cols[0].metric("Sıcaklık Farkı · temp_diff", f"{temp_diff:.2f} K")
    feature_cols[1].metric("Güç Göstergesi · power", f"{power:,.1f}")
    feature_cols[2].metric("Aşınma/Tork · wear_rate", f"{wear_rate:.4f}")
else:
    st.info("Analiz için sol kenar çubuğundan ölçümleri girip **Makine Durumunu Analiz Et** düğmesine basın.")

st.divider()
tab_live, tab_performance = st.tabs(["📡 Canlı Sensör İzleme", "📊 Model Performansı & Özellik Önemi"])

with tab_live:
    st.subheader("Anlık Sensör Ölçümleri")
    live_temp_diff = current_inputs["process_temperature"] - current_inputs["air_temperature"]
    live_power = current_inputs["rotational_speed"] * current_inputs["torque"]
    live_wear_rate = current_inputs["tool_wear"] / (current_inputs["torque"] + 1e-5)
    live_table = pd.DataFrame(
        [
            ("Makine Tipi", current_inputs["Type"], "—"),
            ("Hava Sıcaklığı", current_inputs["air_temperature"], "K"),
            ("Proses Sıcaklığı", current_inputs["process_temperature"], "K"),
            ("Dönme Hızı", current_inputs["rotational_speed"], "rpm"),
            ("Tork", current_inputs["torque"], "Nm"),
            ("Takım Aşınması", current_inputs["tool_wear"], "min"),
            ("temp_diff", live_temp_diff, "K"),
            ("power", live_power, "rpm × Nm"),
            ("wear_rate", live_wear_rate, "min / Nm"),
        ],
        columns=["Ölçüm / Özellik", "Değer", "Birim"],
    )
    st.dataframe(live_table, use_container_width=True, hide_index=True)
    st.caption("Bu özet, kenar çubuğunda seçili olan güncel sensör değerlerini gösterir.")

with tab_performance:
    st.subheader("Test Seti Model Analizleri")
    st.caption("Grafikler `python src/evaluate.py` çalıştırıldıktan sonra oluşturulur.")
    evaluation_dir = PROJECT_ROOT / "outputs" / "evaluation"
    chart_specs = [
        ("feature_importance.png", "Özellik Önemi", "ROC-AUC ile seçilen en iyi modelin tahminde kullandığı göreli özellik katkıları."),
        ("confusion_matrix.png", "Karışıklık Matrisi", "Her iki model için doğru ve hatalı sınıflandırma adetleri."),
        ("roc_curves.png", "ROC Eğrileri", "Modellerin farklı karar eşiklerindeki ayırt etme başarısının karşılaştırması."),
    ]
    chart_cols = st.columns(3)
    for column, (filename, title, description) in zip(chart_cols, chart_specs):
        with column:
            st.markdown(f"#### {title}")
            image_path = evaluation_dir / filename
            if image_path.is_file():
                st.image(str(image_path), use_container_width=True)
                st.caption(description)
            else:
                st.warning(f"Grafik bulunamadı: `{image_path.relative_to(PROJECT_ROOT)}`")
                st.caption(description)
