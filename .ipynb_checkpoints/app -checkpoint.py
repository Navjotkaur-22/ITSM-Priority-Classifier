
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="ITSM Ticket Priority — Inference App", layout="centered")

st.title("ITSM Ticket Priority — Inference App")
st.caption("Loads trained artifacts saved by the notebook and runs predictions.")

BASE = Path(__file__).resolve().parent

def load_artifact(name: str):
    """Try to load artifact from current folder; raise helpful error if missing."""
    path = BASE / name
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {name}. Make sure the notebook saved it to the project root.")
    return joblib.load(path)

# Attempt to load primary classifier (required)
primary_model_name = "priority_rf_pipeline.joblib"
try:
    priority_model = load_artifact(primary_model_name)
    st.success(f"Loaded model: {primary_model_name}")
except Exception as e:
    st.error(f"Could not load primary model '{primary_model_name}'.\n{e}")
    st.stop()

# Optional secondary model (if present in your notebook)
secondary_model = None
secondary_model_name = "rfc_failure_pipeline.joblib"
if (BASE / secondary_model_name).exists():
    try:
        secondary_model = load_artifact(secondary_model_name)
        st.info(f"Optional model also loaded: {secondary_model_name}")
    except Exception:
        pass

st.markdown("### Single Prediction")
with st.form("single"):
    col1, col2 = st.columns(2)

    impact  = col1.selectbox("Impact (1–5)", [1,2,3,4,5], index=1)
    urgency = col2.selectbox("Urgency (1–5)", [1,2,3,4,5], index=1)

    reassign = st.number_input("No_of_Reassignments", min_value=0, value=0, step=1)
    handle   = st.number_input("Handle_Time_hrs", min_value=0.0, value=0.0, step=0.5)

    rel_int = st.number_input("No_of_Related_Interactions", min_value=0, value=0, step=1)
    rel_inc = st.number_input("No_of_Related_Incidents", min_value=0, value=0, step=1)
    rel_chg = st.number_input("No_of_Related_Changes", min_value=0, value=0, step=1)

    status   = st.selectbox("Status", ["Open", "Closed"])
    category = st.text_input("Category", value="incident")
    closure  = st.text_input("Closure_Code", value="Other")

    submitted = st.form_submit_button("Predict")
    if submitted:
        row = pd.DataFrame([{
            "Impact": impact,
            "Urgency": urgency,
            "No_of_Reassignments": reassign,
            "Handle_Time_hrs": handle,
            "No_of_Related_Interactions": rel_int,
            "No_of_Related_Incidents": rel_inc,
            "No_of_Related_Changes": rel_chg,
            "Status": status,
            "Category": category,
            "Closure_Code": closure
        }])
        try:
            pred = priority_model.predict(row)[0]
            st.success(f"Predicted Priority (1–5): **{pred}**")
        except Exception as e:
            st.error(f"Prediction failed. Ensure your model was trained with these feature names.\n{e}")

        if secondary_model is not None:
            try:
                alt_pred = secondary_model.predict(row)[0]
                st.info(f"Secondary model prediction: **{alt_pred}**")
            except Exception:
                pass

st.markdown("---")
st.markdown("### Batch Prediction (CSV)")
st.write("Upload a CSV containing columns exactly like the training set (e.g., Impact, Urgency, No_of_Reassignments, Handle_Time_hrs, No_of_Related_Interactions, No_of_Related_Incidents, No_of_Related_Changes, Status, Category, Closure_Code).")
file = st.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    try:
        dfu = pd.read_csv(file)
        st.write("Preview:", dfu.head())
        preds = priority_model.predict(dfu)
        dfu["Predicted_Priority"] = preds
        st.write("Results (first 10):", dfu.head(10))
        st.download_button("Download Results", dfu.to_csv(index=False), file_name="itsm_predictions.csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
