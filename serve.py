from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np

# Set MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5001")

app = FastAPI(title="Iris Model API")

# Load Production model from MLflow Registry
model = mlflow.sklearn.load_model("models:/IrisBestModel/Production")

# Human-readable class names
class_names = ["setosa", "versicolor", "virginica"]

# Request schema
class IrisRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: IrisRequest):
    """
    Example request:
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    """
    X = np.array([request.features])

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    return {
        "predicted_class": class_names[int(prediction)],
    }