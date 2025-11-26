import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# --- Configuration ---
# In a real setup, set this to your remote Dagshub/MLflow URI
# os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000" 
EXPERIMENT_NAME = "LeadFlow_Scoring_System"
DATA_PATH = "data/leads_data.csv"
MODEL_PATH = "lead_scoring_model.pkl"

def create_synthetic_data(num_samples=1000):
    """Generates synthetic data if the real dataset is missing."""
    print("⚠️ 'data/leads_data.csv' not found. Generating SYNTHETIC data for testing...")
    np.random.seed(42)
    
    data = {
        'Total Time Spent on Website': np.random.randint(0, 2000, num_samples),
        'TotalVisits': np.random.randint(0, 20, num_samples),
        'Lead Source': np.random.choice(['Google', 'Direct Traffic', 'Olark Chat', 'Organic Search'], num_samples),
        'Lead Origin': np.random.choice(['Landing Page Submission', 'API', 'Lead Add Form'], num_samples),
        'Last Activity': np.random.choice(['Email Opened', 'SMS Sent', 'Page Visited on Website', 'Converted to Lead'], num_samples),
        'Converted': np.random.randint(0, 2, num_samples)
    }
    
    # Introduce some basic logic so the model learns patterns
    df = pd.DataFrame(data)
    mask = df['Total Time Spent on Website'] > 1000
    df.loc[mask, 'Converted'] = np.random.choice([0, 1], size=mask.sum(), p=[0.3, 0.7])
    
    os.makedirs('data', exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Synthetic data saved to {DATA_PATH}")
    return df

def build_pipeline():
    """Defines the preprocessing and model pipeline."""
    numeric_features = ["Total Time Spent on Website", "TotalVisits"]
    categorical_features = ["Lead Source", "Lead Origin", "Last Activity"]

    # Handle missing values and scaling for numbers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Handle missing values and encoding for categories
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Combine preprocessing with the Logistic Regression model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', C=0.5, class_weight='balanced'))
    ])
    
    return pipeline

def train():
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 1. Load Data
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Loaded real data from {DATA_PATH}")
    else:
        df = create_synthetic_data()

    # 2. Select Features
    target = "Converted"
    # Ensure these columns exist in your CSV
    features = ["Total Time Spent on Website", "TotalVisits", "Lead Source", "Lead Origin", "Last Activity"]
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. MLflow Tracking
    with mlflow.start_run() as run:
        print(f"🚀 Starting Training Run: {run.info.run_id}")
        
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        # 4. Evaluation
        y_pred = pipeline.predict(X_test)
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0)
        }

        # Log parameters and metrics to MLflow
        mlflow.log_params({"model_type": "LogisticRegression", "class_weight": "balanced"})
        mlflow.log_metrics(metrics)
        
        # Log the complete pipeline model
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Save locally for the FastAPI app to use
        joblib.dump(pipeline, MODEL_PATH)
        
        print("\n📊 Model Performance:")
        for k, v in metrics.items():
            print(f"   - {k}: {v:.4f}")
            
        print(f"\n✅ Model pipeline saved locally to {MODEL_PATH}")

if __name__ == "__main__":
    train()