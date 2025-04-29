from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('penguin_classifier.pkl')

# Print the type of the loaded model
print(f"Loaded model type: {type(model)}")
class PengiunFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    
# Create FastAPI instance
app = FastAPI()

# # Create a POST request endpoint at the route "/predict"
@app.post("/predict")
async def predict_progression(features: PengiunFeatures):
    input_data = pd.DataFrame([features.model_dump()])
    
    prediction = model.predict(input_data)
    return {"predicted_progression": prediction[0]}
