import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Hujan Kolkata - XGBoost",
    page_icon="🌧️",
    layout="wide",
)

sns.set_theme(style="whitegrid")

# Lokasi file mengikuti folder app.py
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dashboard_data.csv"
META_PATH = BASE_DIR / "meta.json"
MODEL_PATH = BASE_DIR / "model_xgb.json"

LABELS = {0: "Tidak Hujan", 1: "Hujan"}


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, parse_dates=["tanggal"])

    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    return data, meta


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


# Memeriksa kelengkapan file
missing_files = [
    path.name
    for path in [DATA_PATH, META_PATH, MODEL_PATH]
    if not path.exists()
]

if missing_files:
    st.error(
        "File berikut belum ditemukan di folder aplikasi: "
        + ", ".join(missing_files)
    )
    st.stop()

data, meta = load_data()
model = load_model()

FEATURES = meta["features"]

# Pembagian data secara kronologis
split_idx = int(len(data) * meta["split_ratio"])

train = data.iloc[:split_idx]
test = data.iloc[split_idx:]

medians = train[FEATURES].median()

X_test = test[FEATURES].fillna(medians)
y_test = test["target"]

# Prediksi data uji
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Evaluasi model
scores = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "F1-Score": f1_score(
        y_test, y_pred, zero_division=0
    ),
}

if y_test.nunique() == 2:
    scores["ROC-AUC"] = roc_auc_score(y_test, y_proba)

# Judul dan deskripsi
st.title("🌧️ Prediksi Hujan Besok di Kolkata")

st.markdown(
    "Dashboard klasifikasi biner menggunakan **XGBoost Classifier** "
    "pada data cuaca harian Kolkata. Model memprediksi apakah "
    "besok hujan berdasarkan kondisi cuaca hari ini, dengan "
    f"ambang curah hujan ≥ {meta['rain_threshold']} mm."
)

# Ringkasan dataset
st.header("1. Ringkasan Dataset")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Jumlah Data", f"{len(data):,}")
c2.metric("Jumlah Fitur", len(FEATURES))
c3.metric(
    "Data Latih / Uji",
    f"{len(train):,} / {len(test):,}",
)
c4.metric(
    "Periode",
    f"{data['tanggal'].min().year}–{data['tanggal'].max().year}",
)

prop = data["target"].map(LABELS).value_counts()

cc1, cc2 = st.columns([1, 2])

with cc1:
    st.write("**Proporsi kelas target**")

    prop_df = pd.DataFrame({
        "Jumlah": prop,
        "Persen": (
            prop / len(data) * 100
        ).round(1).astype(str) + "%",
    })

    st.dataframe(prop_df)

with cc2:
    st.write("**Contoh data**")
    st.dataframe(data.head(10), use_container_width=True)

# Eksplorasi data
st.header("2. Eksplorasi Data (EDA)")

tab1, tab2, tab3 = st.tabs([
    "Distribusi Fitur",
    "Korelasi",
    "Proporsi Kelas",
])

with tab1:
    feat = st.selectbox("Pilih fitur", FEATURES)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.histplot(
        data[feat].dropna(),
        bins=30,
        kde=True,
        ax=ax,
        color="#2E86AB",
    )

    ax.set_title(f"Distribusi {feat}")
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    corr = data[FEATURES + ["target"]].corr()

    size = max(7, 0.5 * len(corr))
    fig, ax = plt.subplots(figsize=(size, size * 0.8))

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        annot=len(corr) <= 15,
        fmt=".2f",
        ax=ax,
    )

    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig, ax = plt.subplots(figsize=(5, 3.5))

    sns.countplot(
        x=data["target"].map(LABELS),
        hue=data["target"].map(LABELS),
        order=["Tidak Hujan", "Hujan"],
        hue_order=["Tidak Hujan", "Hujan"],
        palette={
            "Tidak Hujan": "#F6AE2D",
            "Hujan": "#2E86AB",
        },
        legend=False,
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Jumlah Data")

    st.pyplot(fig)
    plt.close(fig)

# Evaluasi model
st.header("3. Performa Model")

cols = st.columns(len(scores))

for col, (name, value) in zip(cols, scores.items()):
    col.metric(name, f"{value:.3f}")

st.dataframe(
    pd.DataFrame(scores, index=["Nilai"]).T.round(4),
    use_container_width=False,
)

e1, e2 = st.columns(2)

with e1:
    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(figsize=(4.5, 4))

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred, labels=[0, 1]),
        display_labels=["Tidak Hujan", "Hujan"],
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )

    st.pyplot(fig)
    plt.close(fig)

with e2:
    st.subheader("ROC Curve")

    if y_test.nunique() == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba)

        fig, ax = plt.subplots(figsize=(4.5, 4))

        ax.plot(
            fpr,
            tpr,
            color="#2E86AB",
            lw=2,
            label=f"AUC = {scores['ROC-AUC']:.3f}",
        )

        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning(
            "ROC Curve tidak dapat dibuat karena data uji "
            "hanya memiliki satu kelas."
        )

# Feature importance
st.header("4. Feature Importance")

imp = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values().tail(15)

fig, ax = plt.subplots(figsize=(8, 0.4 * len(imp) + 1.5))

ax.barh(imp.index, imp.values, color="#2E86AB")
ax.set_xlabel("Importance")

st.pyplot(fig)
plt.close(fig)

# Prediksi manual
st.header("5. Coba Prediksi Manual")

st.caption(
    "Masukkan kondisi cuaca hari ini. Model akan memprediksi "
    "apakah besok hujan. Nilai awal menggunakan median data latih."
)

with st.form("form_prediksi"):
    inputs = {}
    grid = st.columns(3)

    for i, feature in enumerate(FEATURES):
        valid_values = data[feature].dropna()

        if valid_values.empty:
            st.error(f"Fitur {feature} tidak memiliki data valid.")
            st.stop()

        lo = float(valid_values.min())
        hi = float(valid_values.max())
        default = float(np.clip(medians[feature], lo, hi))

        inputs[feature] = grid[i % 3].number_input(
            feature,
            min_value=lo,
            max_value=hi,
            value=default,
            format="%.2f",
        )

    submitted = st.form_submit_button("Prediksi")

if submitted:
    row = pd.DataFrame([inputs])[FEATURES]

    probability = float(model.predict_proba(row)[0, 1])
    label = int(probability >= 0.5)

    message = (
        f"Prediksi: **{LABELS[label]}** besok "
        f"(probabilitas hujan {probability:.1%})"
    )

    if label == 1:
        st.success(message)
    else:
        st.info(message)

    st.progress(min(max(probability, 0.0), 1.0))