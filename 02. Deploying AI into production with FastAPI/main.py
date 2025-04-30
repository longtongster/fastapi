import numpy as np
from scorer import CommentMetrics, CommentScorer
from fastapi import FastAPI

app = FastAPI()
model = CommentScorer()

@app.post("/predict_trust")
def predict_trust(comment: CommentMetrics):
    # Convert input and extract comment metrics
    features = np.array([[
        comment.length,
        comment.user_reputation,
        comment.report_count
    ]])
    # Get prediction from model 
    score = model.predict(features)
    return {
        "trust_score": round(score, 2),
        "comment_metrics": comment.dict()
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
