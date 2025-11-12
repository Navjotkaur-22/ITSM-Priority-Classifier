# 🧠 ITSM Ticket Priority Classifier — Streamlit App

An end-to-end **Machine Learning project** that predicts ITSM (Incident Ticket Service Management) ticket priority on a scale of **1–5**, using a trained Random Forest classifier.

The app provides both **single record predictions** and **batch CSV scoring**, with an interactive Streamlit dashboard that allows users to visualize and download results easily.

---

[![🚀 Launch App](https://img.shields.io/badge/Launch_Streamlit_App-Click_Here-brightgreen?style=for-the-badge&logo=streamlit)](https://itsm-priority-classifier-zhxrbmmmglzbkuqnyabyt7.streamlit.app/)

---

## ✨ Features
- 🟢 **Single Prediction UI** — user-friendly form for entering incident details  
- 🟣 **Batch Prediction Upload** — upload a CSV file and get predictions instantly  
- 🧩 **Confidence Score Display** — see the model’s certainty for each prediction  
- ⚙️ **Automatic Column Alignment** — handles missing or aliased columns  
- 📥 **Downloadable Results** — one-click export of predictions in CSV format  
- 🌐 **Deployed with Streamlit Cloud**

---

## 📊 Model Overview

- **Algorithm:** Random Forest Classifier  
- **Dataset:** Preprocessed ITSM incident data  
- **Target Variable:** Ticket Priority (1–5)  
- **Training Notebook:** [ITSM_ML_Notebook.ipynb](ITSM_ML_Notebook.ipynb)  
- **Model Artifact:** `priority_rf_pipeline.joblib`  
- **Accuracy:** ~91% (based on cross-validation)  
- **Feature Set:**  
  `Impact`, `Urgency`, `No_of_Reassignments`,  
  `Handle_Time_hrs` / `Resolution_Time_hours`,  
  `No_of_Related_Interactions`, `No_of_Related_Incidents`,  
  `No_of_Related_Changes`, `Status`, `Category`, `Closure_Code`

---

## 🖥 Preview

| Input Form | Batch Upload | Output Table |
|-------------|---------------|---------------|
| ![Input](https://github.com/Navjotkaur-22/ITSM-Priority-Classifier/blob/main/screenshots/form_ui.png) | ![Batch](https://github.com/Navjotkaur-22/ITSM-Priority-Classifier/blob/main/screenshots/batch_upload.png) | ![Output](https://github.com/Navjotkaur-22/ITSM-Priority-Classifier/blob/main/screenshots/output_table.png) |

---

## 🚀 Quick Start (Local Setup)

```bash
# 1️⃣ Create & activate environment
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2️⃣ Install dependencies
pip install -r requirements.txt
# or manually:
pip install streamlit scikit-learn pandas joblib

# 3️⃣ Run Streamlit app
streamlit run app.py
```

---

## 🗂 Project Structure

```
.
├── app.py
├── priority_rf_pipeline.joblib
├── ITSM_ML_Notebook.ipynb
├── requirements.txt
├── sample_data/
│   └── itsm_sample.csv
└── README.md
```

---

## 📄 Batch Prediction Format
Expected CSV header:
```csv
Impact,Urgency,No_of_Reassignments,Resolution_Time_hours,
No_of_Related_Interactions,No_of_Related_Incidents,No_of_Related_Changes,
Status,Category,Closure_Code
```
> The app automatically handles alias columns like `Handle_Time_hrs` → `Resolution_Time_hours`.

---

## 🛠 Tech Stack
| Category | Tools |
|-----------|--------|
| Language | Python 3 |
| Libraries | scikit-learn, pandas, joblib |
| UI | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | GitHub |

---

## 💫 Model Logic Summary

The model predicts ticket priority based on ticket metadata:
- **Impact** and **Urgency** — key drivers for priority
- **Resolution/Handle Time** — numerical severity indicator
- **Related Interactions, Incidents, Changes** — context load
- **Status/Category/Closure_Code** — categorical influences

---

## 📈 Performance Metrics
| Metric | Score |
|---------|-------|
| Accuracy | 91% |
| F1-Score | 0.89 |
| Precision | 0.90 |
| Recall | 0.88 |

---

## 👩‍💻 Author & Links

**Developed by [Navjot Kaur](https://www.upwork.com/freelancers/~01b30aa09d478b524c)**  
💼 *Certified Data Scientist | ML & BI Projects | Streamlit Developer*

<p align="left">
  <a href="https://itsm-priority-classifier-zhxrbmmmglzbkuqnyabyt7.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/Streamlit_App-Open-green?style=for-the-badge&logo=streamlit" />
  </a>
  <a href="https://github.com/Navjotkaur-22/ITSM-Priority-Classifier" target="_blank">
    <img src="https://img.shields.io/badge/GitHub_Repository-Open-blue?style=for-the-badge&logo=github" />
  </a>
  <a href="https://www.upwork.com/freelancers/~01b30aa09d478b524c" target="_blank">
    <img src="https://img.shields.io/badge/Upwork_Profile-View-success?style=for-the-badge&logo=upwork" />
  </a>
</p>

---

## 📬 Contact
📧 Email: *[nkaur4047@gmail.com]*  
🌍 Location: Jalandhar Punjab, India   

---

**© 2025 Navjot Kaur — All Rights Reserved**
