import asyncio
from fastapi import FastAPI, HTTPException
from model import SentimentAnalyzer
from pydantic import BaseModel

app = FastAPI()

class Review(BaseModel):
    text: str

@app.post("/analyze_reviews")
async def analyze_reviews(review: Review):
    try:
        sentiment_model = SentimentAnalyzer()
        # Set model input and timeout limit
        result = await asyncio.wait_for(
            sentiment_model(review.text),
            timeout=10
        )
        return {"sentiment": result["label"]}      
    except asyncio.TimeoutError:
        # Raise HTTP status code for timeout error
        raise HTTPException(status_code=408, detail="Analysis timed out")
    except Exception:
        # Raise HTTP status code for internal error
        raise HTTPException(status_code=500, detail="Analysis failed")
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
