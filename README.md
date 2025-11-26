# 🚀 Project LeadFlow: Intelligent MLOps Lead Scoring System

Project LeadFlow is an end-to-end Machine Learning Operations (MLOps) pipeline designed to optimize sales processes. It predicts the conversion probability of sales leads using a Logistic Regression model, serves predictions via a high-performance FastAPI backend, and visualizes insights on an interactive Streamlit dashboard.

---

## 🏗️ Project Architecture & Functioning

This project follows a microservices-inspired architecture where the Machine Learning lifecycle is decoupled into three distinct stages:

### 1️⃣ The ML Factory – `train.py`

**Role:** Handles data ingestion, preprocessing, and model training.

**Functioning:**

* Checks for the existence of `data/leads_data.csv`. If missing, it automatically generates synthetic data so the code never crashes.
* Builds a scikit-learn Pipeline with:

  * Missing value handling (Imputation)
  * Feature scaling (StandardScaler)
  * Text encoding (OneHotEncoder)
* Trains a Logistic Regression model optimized for high precision.
* Saves the model pipeline as: `lead_scoring_model.pkl`.

### 2️⃣ The Serving Layer – `app.py`

**Role:** A REST API that exposes the model to the world.

**Functioning:**

* Built with FastAPI ⚡ for ultra-fast inference.
* Uses Pydantic for strict input validation.
* Loads `lead_scoring_model.pkl` during startup.
* `/predict` endpoint returns:

  * Score (0–100)
  * Priority (High/Medium/Low)
  * Conversion probability

### 3️⃣ The User Interface – `streamlit_app.py`

**Role:** The frontend dashboard for business users.

**Functioning:**

* Built with Streamlit 🎛️
* Interactive UI for changing lead attributes
* Sends real-time API requests to FastAPI backend
* Gauge chart visualization + strategy alerts (e.g., *"Call within 5 mins"*)

---

## 📂 Project Structure

```
leadflow-mlops/
│
├── 📜 README.md
├── 📜 requirements.txt
├── 📜 train.py
├── 📜 app.py
├── 📜 streamlit_app.py
├── 🐳 Dockerfile
│
├── data/
│   └── leads_data.csv
│
├── .github/
│   └── workflows/
│       └── main.yml
│
└── lead_scoring_model.pkl
```

---

## 🛠️ Installation & Setup Guide

### ✔️ Prerequisites

* Python 3.9+
* Docker Desktop (optional for deployment)

### 🔧 Step-by-Step

```bash
# Create a Virtual Environment
python -m venv venv

# Activate Environment
# Mac/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃‍♂️ Usage Instructions

### Phase 1️⃣: Train the Model

```bash
python train.py
```

➡️ Generates `lead_scoring_model.pkl`

### Phase 2️⃣: Start FastAPI Server

```bash
uvicorn app:app --reload
```

➡️ Visit: **[http://localhost:8000/docs](http://localhost:8000/docs)** (Swagger UI)

### Phase 3️⃣: Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

➡️ Visit: **[http://localhost:8501](http://localhost:8501)**

---

## 🐳 Docker Deployment

Build and run the API in a container:

```bash
docker build -t leadflow-api .
docker run -p 8000:8000 leadflow-api
```

➡️ Test at: **[http://localhost:8000/docs](http://localhost:8000/docs)**

> To use the dashboard with Docker, keep the container running and run Streamlit locally.

---

## 🔌 API Reference

📌 Endpoint: **POST** `/predict`

**Sample Request**

```json
{
  "Total_Time_Spent_on_Website": 600,
  "TotalVisits": 5,
  "Lead_Source": "Google",
  "Lead_Origin": "Landing Page Submission",
  "Last_Activity": "Email Opened"
}
```

**Sample Response**

```json
{
  "score": 85,
  "priority": "High",
  "conversion_probability": 0.8523
}
```

---

## 🧩 Troubleshooting

| Issue                                | Fix                                             |
| ------------------------------------ | ----------------------------------------------- |
| `Module not found`                   | Activate virtual env and reinstall dependencies |
| Dashboard shows `Connection Refused` | Ensure FastAPI server is running                |
| Docker commands fail                 | Make sure Docker Desktop is installed & running |

---

🎯 **Goal:** Enable Sales Teams with actionable AI insights… in real time!

---

💡 *Contributions & improvements are welcome!* 🤝
