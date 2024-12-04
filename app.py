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