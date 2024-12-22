# This is the FastAPI app that gives the predictions

from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
import pickle

# Load the model and scaler
MODEL_PATH = 'housing_price_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI()


# Define a request schema
class HousingData(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: int
    guestroom: int
    basement: int
    hotwaterheating: int
    airconditioning: int
    parking: int
    prefarea: int
    furnishingstatus: int


@app.post("/predict")
def predict(data: HousingData):
    # Convert input data into a NumPy array
    input_data = np.array([[data.area, data.bedrooms, data.bathrooms, data.stories,
                            data.mainroad, data.guestroom, data.basement, data.hotwaterheating,
                            data.airconditioning, data.parking, data.prefarea, data.furnishingstatus]])

    # Predict the price
    prediction = model.predict(input_data)
    return {"predicted_price": float(prediction[0][0])}
