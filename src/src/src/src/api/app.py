"""
FastAPI server for recommendation engine
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from src.inference import recommend

app = FastAPI(
    title="Recommendation Engine API",
    version="1.0.0"
)


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, float]]


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "Recommendation Engine",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend/{user_id}"
        }
    }


@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy"}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: int, k: int = 10):
    """
    Get recommendations for a user
    
    Args:
        user_id: User ID (0-9999)
        k: Number of items (default: 10)
    """
    
    try:
        recs = recommend(user_id, k)
        return {
            "user_id": user_id,
            "recommendations": recs
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Server error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
