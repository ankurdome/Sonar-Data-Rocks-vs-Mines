import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Sonar Rocks vs Mines")

@st.cache_resource
def load_artifacts():
    model = joblib.load("sonar_model.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

model, le = load_artifacts()

st.title("Sonar Rocks vs Mines Classifier")
st.write("Upload a CSV with 60 numeric feature columns (no header), or enter a single sample.")

uploaded = st.file_uploader("Upload CSV (60 columns, no label column)", type=["csv"])

def predict_array(arr):
    X = np.array(arr, dtype=float).reshape(1, -1)
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    return label, proba

if uploaded is not None:
    df = pd.read_csv(uploaded, header=None)
    if df.shape[1] != 60:
        st.error(f"Expected 60 feature columns, found {df.shape[1]}.")
    else:
        preds = model.predict(df.values)
        labels = le.inverse_transform(preds)
        st.subheader("Predictions")
        st.write(pd.DataFrame({"prediction": labels}))
else:
    vals = st.text_area("Enter 60 comma-separated values")
    if st.button("Predict"):
        try:
            arr = [float(x.strip()) for x in vals.split(",")]
            if len(arr) != 60:
                st.error(f"Expected 60 values, got {len(arr)}")
            else:
                label, proba = predict_array(arr)
                st.success(f"Prediction: {label}")
                if proba is not None:
                    st.write({"probabilities": {
                        le.inverse_transform([0])[0]: float(proba[0]),
                        le.inverse_transform([1])[0]: float(proba[1])
                    }})
        except Exception as e:
            st.error(f"Error: {e}")
