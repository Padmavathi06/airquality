import json
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

# Load the model
regmodel = pickle.load(open('air_quality.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

app = FastAPI()

class DataModel(BaseModel):
    data: Dict[str, float]

@app.get("/")
async def home():
    return {"message": "Welcome to the Air Quality Prediction API"}

@app.post("/predict_api")
async def predict_api(data: DataModel):
    new_data = scalar.transform(np.array(list(data.data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    return {"prediction": output[0]}

@app.post("/predict")
async def predict(data: List[float]):
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    output = regmodel.predict(final_input)[0]
    return {"prediction": f"Air Quality {output}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
