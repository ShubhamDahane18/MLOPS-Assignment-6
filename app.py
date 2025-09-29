# app.py
from fastapi import FastAPI
import pickle
from pydantic import BaseModel

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisData):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
