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

# ---------- Harmonization helper (fixes missing / aliased columns) ----------
REQUIRED = [
    "Impact","Urgency","No_of_Reassignments",
    "Handle_Time_hrs","Resolution_Time_hours",   # added Handle_Time_hrs
    "No_of_Related_Interactions","No_of_Related_Incidents","No_of_Related_Changes",
    "Status","Category","Closure_Code"
]

]
ALIASES = {
    "handle_time_hrs": "Resolution_Time_hours",
    "resolution_time_hour": "Resolution_Time_hours",
    "resolution_time_hours": "Resolution_Time_hours",
}

def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    # Add from aliases (case-insensitive)
    lower = {c.lower(): c for c in df.columns}
    for alias, target in ALIASES.items():
        if alias in lower and target not in df.columns:
            df[target] = df[lower[alias]]
    # Ensure both time columns exist by mirroring, if only one is present
    if "Handle_Time_hrs" in df.columns and "Resolution_Time_hours" not in df.columns:
        df["Resolution_Time_hours"] = df["Handle_Time_hrs"]
    if "Resolution_Time_hours" in df.columns and "Handle_Time_hrs" not in df.columns:
        df["Handle_Time_hrs"] = df["Resolution_Time_hours"]

    # Ensure required columns exist; fill neutral defaults
    for col in REQUIRED:
        if col not in df.columns:
            df[col] = 0 if col not in ("Status","Category","Closure_Code") else "Unknown"

    # Keep only known inputs (avoid extra unexpected cols)
    return df[REQUIRED]

# ---------------------------------------------------------------------------

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

    status   = st.selectbox("Status", ["Open", "Resolved", "Closed"])
    category = st.text_input("Category", value="incident")
    closure  = st.text_input("Closure_Code", value="Other")

    submitted = st.form_submit_button("Predict")
    if submitted:
        row = pd.DataFrame([{
            "Impact": impact,
            "Urgency": urgency,
            "No_of_Reassignments": reassign,
            "Handle_Time_hrs": handle,               # kept for aliasing
            "Resolution_Time_hours": handle,         # model expects this — also set explicitly
            "No_of_Related_Interactions": rel_int,
            "No_of_Related_Incidents": rel_inc,
            "No_of_Related_Changes": rel_chg,
            "Status": status,
            "Category": category,
            "Closure_Code": closure
        }])

        try:
            row = harmonize(row)
            pred = priority_model.predict(row)[0]

            # ---- UPGRADE #1: Pretty result badge (+ optional confidence) ----
            conf = None
            try:
                proba = priority_model.predict_proba(row)
                conf = float(proba.max())
            except Exception:
                pass

            priority_map = {1:"1-High", 2:"2-High", 3:"3-Medium", 4:"4-Low", 5:"5-Low"}
            label = priority_map.get(int(pred), str(pred))

            st.markdown(
                f"<div style='padding:10px 14px;border-radius:10px;background:#f0f9ff;"
                f"border:1px solid #bae6fd;display:inline-block;font-size:1.05rem;'>"
                f"<b>Predicted Priority:</b> {label}"
                + (f" &nbsp;•&nbsp; <b>Confidence:</b> {conf:.2f}" if conf is not None else "")
                + "</div>",
                unsafe_allow_html=True
            )
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
st.write("Upload a CSV with these columns (headers can include aliases; app will align them):")
st.code(
    "Impact, Urgency, No_of_Reassignments, Resolution_Time_hours (or Handle_Time_hrs),\n"
    "No_of_Related_Interactions, No_of_Related_Incidents, No_of_Related_Changes,\n"
    "Status, Category, Closure_Code",
    language="text"
)

file = st.file_uploader("Upload CSV", type=["csv"])
if file is not None:
    try:
        dfu = pd.read_csv(file)
        st.write("Preview:", dfu.head())
        dfu = harmonize(dfu)

        preds = priority_model.predict(dfu)
        out = dfu.copy()
        out["Predicted_Priority"] = preds

        st.write("Results (first 10):")
        st.dataframe(out.head(10), use_container_width=True)

        # ---- UPGRADE #2: Clean download button (bytes, no temp file needed) ----
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions (CSV)", data=csv_bytes, file_name="itsm_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
