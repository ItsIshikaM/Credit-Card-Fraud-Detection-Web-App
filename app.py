import io
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🚨",
    layout="wide"
)

# ---------------------------------------------------------
# Load Cached Models
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ---------------------------------------------------------
# Sample Datasets
# (Ensure columns match the exact features your model expects)
# ---------------------------------------------------------
@st.cache_data
def get_sample_datasets():
    # Mixed sample containing both legitimate and high-risk patterns
    mixed_df = pd.DataFrame({
        "Time": [0.0, 15.0, 42.0, 105.0, 120.0],
        "V1": [-1.3598, -3.0435, 1.1918, -2.3122, 1.2345],
        "V2": [-0.0727, 1.7756, 0.2661, 0.4672, -0.2789],
        "V3": [2.5363, -3.1265, 0.1664, 0.8640, 0.5432],
        "V4": [1.3781, 2.9876, 0.4481, -0.1755, -0.1234],
        "Amount": [149.62, 1250.00, 2.69, 378.66, 15.00]
    })

    # High-risk flagged sample
    fraud_df = pd.DataFrame({
        "Time": [406.0, 472.0, 490.0],
        "V1": [-2.3122, -3.0435, -4.1231],
        "V2": [1.9519, 2.1245, 3.4561],
        "V3": [-1.6098, -2.8943, -4.0123],
        "V4": [3.9979, 4.1205, 5.0214],
        "Amount": [0.00, 239.99, 1420.50]
    })
    return {"Mixed Transactions (5 rows)": mixed_df, "High-Risk Transactions (3 rows)": fraud_df}

samples = get_sample_datasets()

# Helper to export dataframe to downloadable CSV
def to_csv_bytes(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")

# ---------------------------------------------------------
# Sidebar: Documentation & Sample Downloads
# ---------------------------------------------------------
with st.sidebar:
    st.header("📂 Sample Data Hub")
    st.markdown(
        "Want to test the file uploader manually? Download the pre-formatted CSVs below:"
    )
    for name, df in samples.items():
        st.download_button(
            label=f"📥 Download {name.split()[0]} CSV",
            data=to_csv_bytes(df),
            file_name=f"{name.split()[0].lower()}_sample.csv",
            mime="text/csv",
            use_container_width=True
        )
    st.divider()
    st.caption("Credit Card Fraud Detection • Production Inference Pipeline")

# ---------------------------------------------------------
# Main Page: Input Selection
# ---------------------------------------------------------
st.title("Credit Card Fraud Detection 🚨")
st.write("Evaluate transactions in real-time. Use built-in test sets or upload your own CSV.")

data_source = st.radio(
    "Choose data source:",
    ["Use built-in sample data (Instant Test)", "Upload your own CSV"],
    horizontal=True
)

active_data = None

if data_source == "Use built-in sample data (Instant Test)":
    selected_sample = st.selectbox("Select sample batch:", list(samples.keys()))
    active_data = samples[selected_sample].copy()
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        active_data = pd.read_csv(uploaded_file)

# ---------------------------------------------------------
# Inference & Results
# ---------------------------------------------------------
if active_data is not None:
    st.subheader("Data Preview")
    st.dataframe(active_data.head(), use_container_width=True)

    # Scaling
    processed_data = active_data.copy()
    if "Amount" in processed_data.columns:
        X = processed_data[["Amount"]].values
        processed_data["Amount"] = scaler.transform(X)
    else:
        st.error("Missing required feature: 'Amount'")
        st.stop()

    model_input = processed_data.select_dtypes(include=["number"])
    if "Class" in model_input.columns:
        model_input = model_input.drop(columns=["Class"])

    # Predict
    try:
        preds = model.predict(model_input)
        
        # Check if probability output is supported
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(model_input)[:, 1]

        active_data["Prediction"] = ["Fraud" if p == 1 else "Legitimate" for p in preds]
        if probs is not None:
            active_data["Fraud Probability"] = [f"{p * 100:.1f}%" for p in probs]

        st.subheader("Results & Analysis")

        fraud_count = (active_data["Prediction"] == "Fraud").sum()
        legit_count = (active_data["Prediction"] == "Legitimate").sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records Evaluated", len(active_data))
        m2.metric("Legitimate", legit_count)
        m3.metric("Flagged Fraud", fraud_count)

        # Highlight fraud rows visually
        def highlight_fraud(row):
            return ["background-color: #ffcccc" if row["Prediction"] == "Fraud" else "" for _ in row]

        st.dataframe(
            active_data.style.apply(highlight_fraud, axis=1),
            use_container_width=True
        )

    # Check exact training feature names
print(model.feature_names_in_)

    except Exception as e:
        st.error(f"Inference error: {e}")
        st.info("Ensure the dataset columns and order match the model's training schema.")
