# ITSM Ticket Priority — Inference App & Project Docs

This repository contains:
- A Jupyter notebook (`ITSM_ML_Notebook.ipynb`) that trains/export models.
- Saved artifacts at the repository **root** (no `outputs/` folder).
- A Streamlit app (`app.py`) to run single/batch predictions using the trained model.

## Artifacts expected (saved by the notebook)
- `priority_rf_pipeline.joblib`  ← primary classifier (required by app)
- `rfc_failure_pipeline.joblib`  ← optional additional model (if your notebook created it)
- `tfidf_vectorizer.joblib`, `nlp_logreg_model.joblib`  ← optional (not used by app)
- `monthly_incident_counts.csv`, `arima_results.joblib` ← optional analytics exports

> Ensure the artifacts are in the **same folder** as `app.py` and `requirements.txt` (repository root).

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- The app builds a form for the typical structured features used in the notebook:
  `Impact, Urgency, No_of_Reassignments, Handle_Time_hrs, No_of_Related_Interactions, No_of_Related_Incidents, No_of_Related_Changes, Status, Category, Closure_Code`
- If your trained pipeline expects a different schema, use the **Batch CSV** uploader with your original headers.
- Model compatibility: keep `scikit-learn` version pinned to avoid joblib/pickle mismatch.
