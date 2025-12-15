# app.py
import os, numpy as np, pandas as pd, joblib
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from tensorflow import keras

THRESHOLD = 0.5
N_SAMPLE = 300
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
RAW_NUMERIC_FIELDS = [ ... ]
FORM_FIELDS = RAW_NUMERIC_FIELDS + CATEGORICAL_COLS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_PATH = os.path.join(BASE_DIR, "UCI_Credit_Card.csv")
TARGET_COL = "default.payment.next.month"

# cache global (serverless: bisa ke-cache antar request selama instance hidup)
_model = None
_cat_imputer = None
_num_imputer = None
_scaler = None
_feature_cols = None
_df_data = None
_dataset_options = None
_has_id = None

def load_everything():
    global _model, _cat_imputer, _num_imputer, _scaler, _feature_cols, _df_data, _dataset_options, _has_id
    if _model is not None:
        return

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

    _df_data = pd.read_csv(DATA_PATH)
    if TARGET_COL not in _df_data.columns:
        raise ValueError(f"Kolom target '{TARGET_COL}' tidak ditemukan.")

    _has_id = "ID" in _df_data.columns

    required = [
        "ann_credit_default.keras",
        "cat_imputer.joblib",
        "num_imputer.joblib",
        "scaler.joblib",
        "feature_columns.joblib",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(ARTIFACT_DIR, f))]
    if missing:
        raise FileNotFoundError("Artifacts belum lengkap: " + ", ".join(missing))

    _model = keras.models.load_model(os.path.join(ARTIFACT_DIR, "ann_credit_default.keras"))
    _cat_imputer = joblib.load(os.path.join(ARTIFACT_DIR, "cat_imputer.joblib"))
    _num_imputer = joblib.load(os.path.join(ARTIFACT_DIR, "num_imputer.joblib"))
    _scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib"))
    _feature_cols = joblib.load(os.path.join(ARTIFACT_DIR, "feature_columns.joblib"))

    df_sample = _df_data.head(N_SAMPLE).copy()
    if _has_id:
        _dataset_options = [{"row_key": int(i), "display": f"Baris #{int(i)+1} (ID={int(r['ID'])})"} for i, r in df_sample.iterrows()]
    else:
        _dataset_options = [{"row_key": int(i), "display": f"Baris #{int(i)+1}"} for i in df_sample.index]

def create_app():
    app = Flask(__name__)
    app.secret_key = "ta11-credit-default"

    def label_text(y: int) -> str:
        return "GAGAL BAYAR" if int(y) == 1 else "TIDAK GAGAL"

    def row_to_form_values(row: pd.Series) -> dict:
        return {c: row[c] if c in row.index else 0 for c in FORM_FIELDS}

    def preprocess_one(df_raw_one: pd.DataFrame) -> np.ndarray:
        load_everything()
        df_imp = df_raw_one.copy()

        numeric_cols = [c for c in df_imp.columns if c not in CATEGORICAL_COLS]
        for c in numeric_cols:
            df_imp[c] = pd.to_numeric(df_imp[c], errors="coerce")
        for c in CATEGORICAL_COLS:
            if c in df_imp.columns:
                df_imp[c] = pd.to_numeric(df_imp[c], errors="coerce")

        if "EDUCATION" in df_imp.columns:
            df_imp["EDUCATION"] = df_imp["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
        if "MARRIAGE" in df_imp.columns:
            df_imp["MARRIAGE"] = df_imp["MARRIAGE"].replace({0: 3})

        present_cat = [c for c in CATEGORICAL_COLS if c in df_imp.columns]
        if present_cat:
            df_imp[present_cat] = _cat_imputer.transform(df_imp[present_cat])
        if numeric_cols:
            df_imp[numeric_cols] = _num_imputer.transform(df_imp[numeric_cols])

        df_enc = pd.get_dummies(df_imp, columns=present_cat, drop_first=False)
        df_enc = df_enc.reindex(columns=_feature_cols, fill_value=0)
        x_scaled = _scaler.transform(df_enc)
        return x_scaled

    @app.route("/", methods=["GET"])
    def home():
        load_everything()
        return render_template("index.html", result=None, values={}, dataset_options=_dataset_options, selected_row="")

    @app.route("/predict", methods=["POST"])
    def predict():
        load_everything()
        row_key = request.form.get("dataset_row", "").strip()
        if row_key == "":
            flash("Silakan pilih data dari dataset terlebih dahulu.")
            return redirect(url_for("home"))

        row_key = int(float(row_key))
        if row_key < 0 or row_key >= len(_df_data):
            flash("Pilihan data dataset tidak valid.")
            return redirect(url_for("home"))

        row = _df_data.iloc[row_key]
        true_y = int(row[TARGET_COL])
        true_label = label_text(true_y)

        form_values = row_to_form_values(row)
        df_one = pd.DataFrame([form_values])
        x = preprocess_one(df_one)

        prob = float(_model.predict(x, verbose=0).ravel()[0])
        model_pred = 1 if prob >= THRESHOLD else 0
        model_label = label_text(model_pred)

        result = {
            "row_key": row_key,
            "true_y": true_y,
            "true_label": true_label,
            "prob": prob,
            "model_pred": model_pred,
            "model_label": model_label,
            "threshold": THRESHOLD,
            "is_correct": int(model_pred == true_y),
        }
        if _has_id:
            result["id"] = int(row["ID"])

        values = {k: str(form_values.get(k, "")) for k in FORM_FIELDS}
        return render_template("index.html", result=result, values=values, dataset_options=_dataset_options, selected_row=str(row_key))

    @app.route("/health", methods=["GET"])
    def health():
        load_everything()
        return jsonify({
            "status": "ok",
            "dataset_rows": int(_df_data.shape[0]),
            "sample_rows": int(min(N_SAMPLE, _df_data.shape[0])),
            "threshold": THRESHOLD,
            "features_after_ohe": int(len(_feature_cols)),
            "has_id_column": bool(_has_id),
        })

    return app

app = create_app()
