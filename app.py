import os
import numpy as np
import pandas as pd
import joblib

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from tensorflow import keras

app = Flask(__name__)
app.secret_key = "ta11-credit-default"

# =========================
# KONFIGURASI
# =========================
THRESHOLD = 0.5          # bisa kamu ubah jadi 0.6 kalau false positive terlalu banyak
N_SAMPLE = 300           # jumlah baris dataset yang ditampilkan di dropdown (biar ringan)

CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]

RAW_NUMERIC_FIELDS = [
    "LIMIT_BAL", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

FORM_FIELDS = RAW_NUMERIC_FIELDS + CATEGORICAL_COLS


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_PATH = os.path.join(BASE_DIR, "UCI_Credit_Card.csv")


# =========================
# A) LOAD DATASET (GROUND TRUTH)
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

df_data = pd.read_csv(DATA_PATH)

TARGET_COL = "default.payment.next.month"
if TARGET_COL not in df_data.columns:
    raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan di dataset.")

# Drop ID dari logika model (ID tidak dipakai model), tapi boleh dipakai untuk display
HAS_ID = "ID" in df_data.columns


# =========================
# B) LOAD ARTIFACTS ANN
# =========================
REQUIRED_FILES = [
    "ann_credit_default.keras",
    "cat_imputer.joblib",
    "num_imputer.joblib",
    "scaler.joblib",
    "feature_columns.joblib",
]

missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(ARTIFACT_DIR, f))]
if missing:
    raise FileNotFoundError(
        "Artifacts belum lengkap, file berikut tidak ditemukan:\n- "
        + "\n- ".join(missing)
        + f"\n\nPastikan semua file ada di folder: {ARTIFACT_DIR}"
    )

model = keras.models.load_model(os.path.join(ARTIFACT_DIR, "ann_credit_default.keras"))
cat_imputer = joblib.load(os.path.join(ARTIFACT_DIR, "cat_imputer.joblib"))
num_imputer = joblib.load(os.path.join(ARTIFACT_DIR, "num_imputer.joblib"))
scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))
feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "feature_columns.joblib"))


# =========================
# UTIL
# =========================
def label_text(y: int) -> str:
    # mapping FINAL: 1 = default (gagal bayar), 0 = tidak default
    return "GAGAL BAYAR" if int(y) == 1 else "TIDAK GAGAL"


def row_to_form_values(row: pd.Series) -> dict:
    """Ambil kolom yang dipakai form dari 1 baris dataset."""
    values = {}
    for c in FORM_FIELDS:
        values[c] = row[c] if c in row.index else 0
    return values


def preprocess_one(df_raw_one: pd.DataFrame) -> np.ndarray:
    """
    Pipeline harus SAMA dengan training:
    - pastikan dtype numeric konsisten
    - cleaning kategori (EDUCATION, MARRIAGE) SAMA
    - imputasi kategori & numerik (artifact)
    - one-hot -> reindex -> scaling
    """
    df_imp = df_raw_one.copy()

    # 1) pastikan numerik benar-benar numeric
    numeric_cols = [c for c in df_imp.columns if c not in CATEGORICAL_COLS]
    for c in numeric_cols:
        df_imp[c] = pd.to_numeric(df_imp[c], errors="coerce")

    # 2) pastikan kategori numeric (karena imputernya dilatih dari angka)
    for c in CATEGORICAL_COLS:
        if c in df_imp.columns:
            df_imp[c] = pd.to_numeric(df_imp[c], errors="coerce")

    # 3) cleaning kategori harus sama seperti training
    if "EDUCATION" in df_imp.columns:
        df_imp["EDUCATION"] = df_imp["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    if "MARRIAGE" in df_imp.columns:
        df_imp["MARRIAGE"] = df_imp["MARRIAGE"].replace({0: 3})

    present_cat = [c for c in CATEGORICAL_COLS if c in df_imp.columns]

    # 4) imputasi kategori
    if present_cat:
        df_imp[present_cat] = cat_imputer.transform(df_imp[present_cat])

    # 5) imputasi numerik
    if numeric_cols:
        df_imp[numeric_cols] = num_imputer.transform(df_imp[numeric_cols])

    # 6) one-hot
    df_enc = pd.get_dummies(df_imp, columns=present_cat, drop_first=False)

    # 7) samakan kolom dengan training
    df_enc = df_enc.reindex(columns=feature_cols, fill_value=0)

    # 8) scaling
    x_scaled = scaler.transform(df_enc)
    return x_scaled


# =========================
# DROPDOWN OPTIONS
# =========================
df_sample = df_data.head(N_SAMPLE).copy()

if HAS_ID:
    DATASET_OPTIONS = [
        {"row_key": int(idx), "display": f"Baris #{int(idx)+1} (ID={int(row['ID'])})"}
        for idx, row in df_sample.iterrows()
    ]
else:
    DATASET_OPTIONS = [
        {"row_key": int(idx), "display": f"Baris #{int(idx)+1}"}
        for idx in df_sample.index
    ]


# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        result=None,
        values={},
        dataset_options=DATASET_OPTIONS,
        selected_row=""
    )


@app.route("/row/<int:row_key>", methods=["GET"])
def get_row(row_key: int):
    """Return 1 baris dataset sebagai JSON untuk auto-fill form (kalau dipakai JS)."""
    if row_key < 0 or row_key >= len(df_data):
        return jsonify({"ok": False, "error": "row_key tidak valid"}), 400

    row = df_data.iloc[row_key]
    values = row_to_form_values(row)
    true_y = int(row[TARGET_COL])

    resp = {
        "ok": True,
        "row_key": row_key,
        "values": values,
        "true_y": true_y,
        "true_label": label_text(true_y)
    }

    if HAS_ID:
        resp["id"] = int(row["ID"])

    return jsonify(resp)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil row_key dari dropdown
        row_key = request.form.get("dataset_row", "").strip()
        if row_key == "":
            flash("Silakan pilih data dari dataset terlebih dahulu.")
            return redirect(url_for("home"))

        row_key = int(float(row_key))
        if row_key < 0 or row_key >= len(df_data):
            flash("Pilihan data dataset tidak valid.")
            return redirect(url_for("home"))

        row = df_data.iloc[row_key]

        # Ground truth dari dataset
        true_y = int(row[TARGET_COL])
        true_label = label_text(true_y)

        # Input model dari dataset (tanpa target, tanpa ID)
        form_values = row_to_form_values(row)
        df_one = pd.DataFrame([form_values])

        x = preprocess_one(df_one)

        prob = float(model.predict(x, verbose=0).ravel()[0])
        model_pred = 1 if prob >= THRESHOLD else 0
        model_label = label_text(model_pred)

        is_correct = int(model_pred == true_y)   # 1 benar, 0 salah

        result = {
            "row_key": row_key,
            "true_y": true_y,
            "true_label": true_label,
            "prob": prob,
            "model_pred": model_pred,
            "model_label": model_label,
            "threshold": THRESHOLD,
            "is_correct": is_correct,
        }

        if HAS_ID:
            result["id"] = int(row["ID"])

        # supaya form tetap tampil nilai yang dipilih (kalau kamu masih tampilkan input)
        values = {k: str(form_values.get(k, "")) for k in FORM_FIELDS}

        return render_template(
            "index.html",
            result=result,
            values=values,
            dataset_options=DATASET_OPTIONS,
            selected_row=str(row_key)
        )

    except Exception as e:
        flash(f"Error: {str(e)}")
        return redirect(url_for("home"))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "dataset_rows": int(df_data.shape[0]),
        "sample_rows": int(min(N_SAMPLE, df_data.shape[0])),
        "threshold": THRESHOLD,
        "features_after_ohe": int(len(feature_cols)),
        "has_id_column": bool(HAS_ID),
    })


if __name__ == "__main__":
    app.run(debug=True)