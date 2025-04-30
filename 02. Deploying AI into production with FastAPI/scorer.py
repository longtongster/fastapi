import numpy as np
from pydantic import BaseModel

class CommentMetrics(BaseModel):
    length: int
    user_reputation: int
    report_count: int

class CommentScorer:
    def predict(self, features: np.ndarray) -> float:
        """
        Predict trust score based on comment metrics
        features: [[length, user_reputation, report_count]]
        """
        # Unpack features
        length, reputation, reports = features[0]
        
        # Calculate trust score
        score = (0.3 * (length/500) +        # Normalize length
                 0.5 * (reputation/100) +    # Normalize reputation
                 -0.2 * reports)             # Reports reduce score
        
        return float(max(min(score * 100, 100), 0))  # Scale to 0-100
