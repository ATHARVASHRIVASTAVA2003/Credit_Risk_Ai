
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model['model'] = joblib.load(BASE_DIR / 'credit_risk_model.pkl')
    ml_model['threshold'] = joblib.load(BASE_DIR / 'best_threshold.pkl')

    yield

    ml_model.clear()


app = FastAPI(lifespan=lifespan)


# The only columns that user will see and provide inputs.
class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


@app.post('/predict')
def predict(data: LoanApplication):

    print("Received prediction request")

    input_df = pd.DataFrame([data.dict()])

    print("Input Data:")
    print(input_df)

    print("Model:", type(ml_model['model']))
    print("Threshold:", ml_model["threshold"])
    print("Threshold Type:", type(ml_model["threshold"]))

    probability = ml_model['model'].predict_proba(input_df)[:, 1][0]

    print("Probability:", probability)

    # Risk classification
    if probability < 0.30:
        risk_level = "LOW RISK"
    elif probability < 0.70:
        risk_level = "MODERATE RISK"

    else:
        risk_level = "HIGH RISK"

    prediction = int(probability >= ml_model["threshold"])

    print("Prediction:", prediction)

    return {
        "default_probability": float(probability),
        "default_prediction": prediction,
        "threshold": float(ml_model["threshold"]),
        "Result": "High Risk" if prediction == 1 else "Low Risk",
        "risk_level": risk_level
    }


# Serve original Credit Ledger frontend
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
