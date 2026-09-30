import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🚨",
    layout="wide"
)

# ---------------------------------------------------------
# Load Cached Model and Scaler
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# Inspect required model features
if hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    # Standard Kaggle credit card dataset feature ordering
    expected_features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ---------------------------------------------------------
# Realistic Pre-loaded Test Samples (All 30 Features)
# ---------------------------------------------------------
@st.cache_data
def get_sample_datasets():
    # Legitimate transaction row (PCA values close to zero)
    legit_tx = {
        "Time": 0.0,
        "Amount": 149.62,
        **{f"V{i}": round(np.random.normal(0, 0.2), 4) for i in range(1, 29)}
    }
    legit_tx.update({"V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781})

    # Fraudulent transaction row (anomalous PCA distributions)
    fraud_tx = {
        "Time": 406.0,
        "Amount": 245.00,
        **{f"V{i}": round(np.random.normal(0, 0.5), 4) for i in range(1, 29)}
    }
    fraud_tx.update({
        "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979,
        "V10": -3.7600, "V12": -4.8999, "V14": -5.2892, "V17": -3.8300
    })

    mixed_df = pd.DataFrame([legit_tx, fraud_tx, legit_tx])
    mixed_df = mixed_df.reindex(columns=expected_features)

    fraud_df = pd.DataFrame([fraud_tx, fraud_tx])
    fraud_df = fraud_df.reindex(columns=expected_features)

    return {
        "Mixed Sample (Legitimate & Fraud)": mixed_df,
        "High-Risk Sample (Flagged Cases)": fraud_df
    }

samples = get_sample_datasets()

def to_csv_bytes(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("📂 Sample Data Hub")
    st.markdown("Download pre-formatted test CSV files containing all 30 features:")
    for name, df in samples.items():
        st.download_button(
            label=f"📥 Download {name.split()[0]} CSV",
            data=to_csv_bytes(df),
            file_name=f"{name.split()[0].lower()}_test.csv",
            mime="text/csv",
            use_container_width=True
        )

    st.divider()

    # Kaggle Dataset Link
    st.subheader("🌐 Source Dataset")
    st.markdown(
        """
        Trained on the benchmark **Credit Card Fraud Detection** dataset.
        
        👉 [View on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
        """
    )
    st.link_button(
        label="Open Kaggle Dataset ↗",
        url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        use_container_width=True
    )

    st.divider()
    st.info(
        f"**Expected Inputs:** {len(expected_features)} features "
        f"(`Time`, `V1`–`V28`, `Amount`)."
    )

# ---------------------------------------------------------
# Main Page: Data Selection
# ---------------------------------------------------------
st.title("Credit Card Fraud Detection 🚨")
st.markdown(
    "Detect fraudulent transactions in real-time. Use the pre-loaded benchmark sets or test with your own CSV."
)

data_source = st.radio(
    "Select Data Source:",
    ["Use Pre-loaded Sample Data (Recommended)", "Upload CSV File"],
    horizontal=True
)

active_data = None

if data_source == "Use Pre-loaded Sample Data (Recommended)":
    selected_sample = st.selectbox("Choose sample batch:", list(samples.keys()))
    active_data = samples[selected_sample].copy()
else:
    uploaded_file = st.file_uploader(
        "Upload transaction CSV (must contain features or download template from sidebar)",
        type=["csv"]
    )
    if uploaded_file is not None:
        active_data = pd.read_csv(uploaded_file)

# ---------------------------------------------------------
# Processing & Inference Pipeline
# ---------------------------------------------------------
if active_data is not None:
    st.subheader("Data Preview")
    st.dataframe(active_data.head(), use_container_width=True)

    # 1. Feature check & Graceful Fallback
    missing_cols = [c for c in expected_features if c not in active_data.columns]
    
    if missing_cols:
        st.warning(
            f"⚠️ Your CSV is missing {len(missing_cols)} expected columns (e.g. {missing_cols[:4]}...).\n\n"
            "The app has auto-imputed missing PCA features with default baseline values `0.0` so you can still test inference."
        )
        for col in missing_cols:
            active_data[col] = 0.0

    # Ensure required Amount column exists
    if "Amount" not in active_data.columns:
        st.error("Error: The dataset must have at least an 'Amount' column.")
        st.stop()

    # 2. Prepare data for model
    inference_df = active_data.copy()

    # Scale the Amount feature
    inference_df["Amount"] = scaler.transform(inference_df[["Amount"]].values)

    # Reorder and select strictly the expected features
    model_input = inference_df[expected_features]

    # 3. Predict
    try:
        preds = model.predict(model_input)
        
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(model_input)[:, 1]

        active_data["Prediction"] = ["Fraud" if p == 1 else "Legitimate" for p in preds]
        if probs is not None:
            active_data["Risk Score"] = [f"{p * 100:.1f}%" for p in probs]

        st.subheader("Classification Results")

        legit_count = (active_data["Prediction"] == "Legitimate").sum()
        fraud_count = (active_data["Prediction"] == "Fraud").sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Evaluated Records", len(active_data))
        m2.metric("Legitimate", legit_count)
        m3.metric("Fraudulent Flagged", fraud_count)

        # Highlight fraud rows in soft red
        def highlight_fraud(row):
            return ["background-color: #ffd2d2" if row["Prediction"] == "Fraud" else "" for _ in row]

        st.dataframe(
            active_data.style.apply(highlight_fraud, axis=1),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Inference could not be executed: {e}")
