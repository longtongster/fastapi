# File: sentiment_api.py
import joblib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# model creation
def train_and_save_model():
    data = {
        "review": [
            "I love this product, it's fantastic!",
            "Really satisfied with the quality!",
            "Terrible, I hate it.",
            "Not happy with the purchase.",
            "Absolutely amazing and wonderful!",
            "Worst experience ever.",
            "I am very pleased with my purchase.",
            "Disappointed, it didn't work as expected.",
            "The best thing I've ever bought.",
            "Totally awful, will not buy again."
        ],
        "label": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
    }

    df = pd.DataFrame(data)

    positive_words = ["love", "satisfied", "amazing", "fantastic", "wonderful", "pleased", "best"]
    negative_words = ["hate", "terrible", "worst", "disappointed", "awful"]

    df["num_words"] = df["review"].apply(lambda x: len(x.split()))
    df["num_positive_words"] = df["review"].apply(lambda x: sum(word in x.lower() for word in positive_words))
    df["num_complaints"] = df["review"].apply(lambda x: sum(word in x.lower() for word in negative_words))

    X = df[["num_words", "num_positive_words", "num_complaints"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # Save model using joblib instead of pickle
    joblib.dump(model, 'sentiment_model.joblib', compress=3)

# Train and save the model
train_and_save_model()

sentiment_model = None

# Define a callable class
class SentimentAnalyzer:
    def __init__(self, model_path):
        # Load the model using joblib
        self.model = joblib.load(model_path)
        self.positive_words = ["love", "satisfied", "amazing", "fantastic", "wonderful", "pleased", "best"]
        self.negative_words = ["hate", "terrible", "worst", "disappointed", "awful"]

    def __call__(self, text):
        num_words = len(text.split())
        num_positive_words = sum(word in text.lower() for word in self.positive_words)
        num_complaints = sum(word in text.lower() for word in self.negative_words)
        features = [[num_words, num_positive_words, num_complaints]]
        
        # Get prediction and confidence score
        prediction = self.model.predict(features)
        confidence_scores = self.model.predict_proba(features)
        
        # Return dictionary directly instead of JSON string
        result = {
            "label": "Positive" if prediction[0] == 1 else "Negative",
            "confidence": float(confidence_scores[0][prediction[0]])  
        }
        return result 

# Define model loading function
def load_model():
    global sentiment_model
    sentiment_model = SentimentAnalyzer("sentiment_model.joblib")

# Initialize FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize our mock model on startup
    global sentiment_model
    load_model()
    yield


# Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

# Initialize model variable
sentiment_model = None

# Define request/response models
class CommentRequest(BaseModel):
    text: str

class CommentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    

@app.post("/analyze")
def analyze_comment(request: CommentRequest):
    if sentiment_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Empty text provided"
        )
        
    result = sentiment_model(request.text)
    return CommentResponse(
        text=request.text,
        sentiment=result["label"],
        confidence=result["confidence"]
    )

@app.get("/health")
def health_check():
    """Check if model is loaded and ready"""
    return {
        "status": "healthy" if sentiment_model is not None else "unhealthy",
        "model_loaded": sentiment_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
