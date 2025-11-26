from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# --- Configurations ---
MODEL_PATH = "lead_scoring_model.pkl"

# --- Schema Definition (Pydantic) ---
# This ensures that the API only accepts valid data types
class LeadInput(BaseModel):
    Total_Time_Spent_on_Website: int
    TotalVisits: int
    Lead_Source: str
    Lead_Origin: str
    Last_Activity: str

    class Config:
        json_schema_extra = {
            "example": {
                "Total_Time_Spent_on_Website": 600,
                "TotalVisits": 5,
                "Lead_Source": "Google",
                "Lead_Origin": "Landing Page Submission",
                "Last_Activity": "Email Opened"
            }
        }

class PredictionOutput(BaseModel):
    score: int
    priority: str
    conversion_probability: float

# --- App Initialization ---
app = FastAPI(
    title="LeadFlow API",
    description="Predictive Lead Scoring Engine",
    version="1.0"
)

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        else:
            print("⚠️ Model file not found. Please run train.py first.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict_lead(lead: LeadInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run training first.")
    
    # Convert Pydantic object to DataFrame
    # Column names must match exactly what was used in train.py
    data = {
        "Total Time Spent on Website": [lead.Total_Time_Spent_on_Website],
        "TotalVisits": [lead.TotalVisits],
        "Lead Source": [lead.Lead_Source],
        "Lead Origin": [lead.Lead_Origin],
        "Last Activity": [lead.Last_Activity]
    }
    df_input = pd.DataFrame(data)

    try:
        # Get probability of class 1 (Converted)
        probability = model.predict_proba(df_input)[0][1]
        score = int(probability * 100)

        # Determine priority based on score
        if score >= 80:
            priority = "High"
        elif score >= 50:
            priority = "Medium"
        else:
            priority = "Low"

        return {
            "score": score,
            "priority": priority,
            "conversion_probability": round(probability, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")