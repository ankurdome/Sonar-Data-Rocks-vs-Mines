# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Sonar Rocks vs Mines", page_icon="🌊", layout="wide")

# ---- Load artifacts ----
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("sonar_model.pkl")   # sklearn Pipeline with scaler + classifier
        le = joblib.load("label_encoder.pkl")    # LabelEncoder for ['R','M']
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()
    return model, le

model, le = load_artifacts()

st.title("Sonar Rocks vs Mines Classifier")
st.write("Upload CSV with either 60 features only, or the original Sonar file with 60 features + trailing R/M label.")

# ---- Helpers ----
ZERO_WIDTH_PATTERN = re.compile(r'[\u200B-\u200D\uFEFF]')

def strip_zero_width(s: str) -> str:
    return ZERO_WIDTH_PATTERN.sub("", s).strip()

def clean_df_zero_width(df: pd.DataFrame) -> pd.DataFrame:
    # Remove zero-width and BOM characters from every cell (strings only)
    return df.applymap(lambda x: strip_zero_width(x) if isinstance(x, str) else x)

def prepare_features(df_raw: pd.DataFrame):
    """
    Returns:
      df_num: numeric features DataFrame with exactly 60 columns
      y_true: np.ndarray of encoded labels if present, else None
      y_true_str: np.ndarray of string labels ['R','M'] if present, else None
    """
    # Remove zero-width chars possibly introduced by copy-paste
    df_raw = clean_df_zero_width(df_raw)

    # Try to preserve a possible label column before coercion
    y_true_str = None
    if df_raw.shape[1] >= 61:
        last_col = df_raw.iloc[:, -1]
        if last_col.dtype == object:
            y_true_str = last_col.astype(str).values

    # Coerce to numeric; label column becomes NaN
    df_num = df_raw.apply(pd.to_numeric, errors="coerce")

    # If 61 columns and last becomes all NaN (from R/M), drop it
    if df_num.shape[1] == 61 and df_num.iloc[:, -1].isna().all():
        df_num = df_num.iloc[:, :-1]

    # If there are more than 60 columns, keep the first 60
    if df_num.shape[1] > 60:
        df_num = df_num.iloc[:, :60]

    # Encode labels if we detected R/M
    y_true = None
    if y_true_str is not None and df_num.shape[1] == 60:
        try:
            y_true = le.transform(y_true_str)  # requires same encoder as training
        except Exception:
            y_true = None

    return df_num, y_true, y_true_str

def predict_array(arr):
    X = np.array(arr, dtype=float).reshape(1, -1)
    pred = model.predict(X)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            proba = None
    label = le.inverse_transform([pred])[0]
    return label, proba

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig

def plot_roc(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return fig

def compute_scores_for_roc(X):
    # Prefer predict_proba, else use decision_function if available
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            dec = model.decision_function(X)
            if dec.ndim == 1:
                return dec
            elif dec.ndim == 2 and dec.shape[1] == 2:
                return dec[:, 1]
        except Exception:
            pass
    return None

# ---- Upload and predict ----
uploaded = st.file_uploader("Upload CSV (60 features, or 60+label R/M)", type=["csv"])

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded, header=None)
        df_num, y_true, y_true_str = prepare_features(raw)

        # Validate numeric shape/values (StandardScaler requires finite numbers)
        if df_num.shape[1] != 60:
            st.error(f"Expected 60 feature columns after cleaning, found {df_num.shape[1]}.")
        elif df_num.isna().any().any() or np.isinf(df_num.to_numpy()).any():
            bad_rows = df_num.index[df_num.isna().any(axis=1)].tolist()
            st.error(f"Non-numeric or missing values in rows: {bad_rows}. Please clean your CSV.")
        else:
            X = df_num.values
            preds = model.predict(X)
            labels_pred = le.inverse_transform(preds)

            st.subheader("Predictions")
            st.dataframe(pd.DataFrame({"prediction": labels_pred}))

            # ===== EDA section =====
            st.markdown("### EDA")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.caption("Feature histogram")
                feat_idx = st.number_input("Feature index (0-59)", min_value=0, max_value=59, value=0, step=1)
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.hist(df_num.iloc[:, int(feat_idx)], bins=20, color="#2c7fb8", alpha=0.9)
                ax.set_title(f"Feature {int(feat_idx)} distribution")
                ax.set_xlabel("Value")
                ax.set_ylabel("Count")
                st.pyplot(fig, use_container_width=True)

            with c2:
                st.caption("Correlation heatmap (subset)")
                n_heat = st.slider("Columns for heatmap", min_value=5, max_value=60, value=20, step=5)
                corr = df_num.iloc[:, :n_heat].corr()
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
                ax.set_title(f"Correlation heatmap (first {n_heat} features)")
                st.pyplot(fig, use_container_width=True)

            with c3:
                st.caption("PCA (2D) scatter")
                try:
                    X_scaled = StandardScaler().fit_transform(X)
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    color_choice = st.radio("Color by", ["Predicted label", "Actual label (if present)"], index=0)
                    if color_choice.startswith("Actual") and y_true is not None:
                        hue = y_true
                        legend_labels = le.inverse_transform([0, 1])
                    else:
                        hue = preds
                        legend_labels = le.inverse_transform([0, 1])

                    fig, ax = plt.subplots(figsize=(4, 3))
                    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=hue, cmap="coolwarm", alpha=0.8)
                    ax.set_title("PCA (2 components)")
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    handles = [plt.Line2D([0], [0], marker='o', color='w',
                                          label=legend_labels[i], markerfacecolor=plt.cm.coolwarm([0,1][i]), markersize=8)
                               for i in range(2)]
                    ax.legend(handles=handles, title="Class", loc="best")
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"PCA plot unavailable: {e}")

            # ===== Evaluation (when labels available) =====
            st.markdown("### Evaluation (requires ground-truth labels)")
            if y_true is not None:
                col_a, col_b = st.columns(2)

                with col_a:
                    st.caption("Confusion matrix")
                    fig = plot_confusion_matrix(y_true, preds, labels=le.inverse_transform([0, 1]))
                    st.pyplot(fig, use_container_width=True)

                with col_b:
                    st.caption("ROC curve")
                    scores = compute_scores_for_roc(X)
                    if scores is not None:
                        fig = plot_roc(y_true, scores)
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("Classifier does not expose probabilities or decision scores; ROC unavailable.")
            else:
                st.info("Ground-truth labels not detected; showing EDA and predictions only.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

# ---- Manual single-sample path ----
st.markdown("---")
st.subheader("Or paste a single sample")
vals = st.text_area("Enter 60 comma-separated values (e.g., 0.02,0.0371,...)", height=100)

if st.button("Predict"):
    try:
        tokens = [strip_zero_width(x) for x in vals.split(",")]
        arr = [float(t) for t in tokens if t != ""]
        if len(arr) != 60:
            st.error(f"Expected 60 values, got {len(arr)}.")
        else:
            label, proba = predict_array(arr)
            st.success(f"Prediction: {label}")
            if proba is not None and len(proba) == 2:
                classes = le.inverse_transform([0, 1])
                st.write({"probabilities": {classes[0]: float(proba[0]), classes[1]: float(proba[1])}})
    except Exception as e:
        st.error(f"Error: {e}")

# ---- Sidebar guidance ----
with st.sidebar:
    st.caption("Input format")
    st.write("- 60 numeric columns only for inference; if your CSV has the R/M label as the 61st column, it will be auto-dropped during cleaning.")
    st.write("- No header row is needed; values must be finite numbers.")
