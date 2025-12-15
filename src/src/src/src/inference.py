"""
Inference module for generating recommendations
"""

import numpy as np
from tensorflow import keras
from config import *


# Load model
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except:
    model = None
    print("⚠️  Model not found. Run training first.")


def recommend(user_id: int, k: int = 10):
    """
    Get top-K recommendations for a user
    
    Args:
        user_id: User ID
        k: Number of recommendations
    
    Returns:
        List of recommendations with scores
    """
    
    if model is None:
        raise RuntimeError("Model not loaded")
    
    if user_id < 0 or user_id >= N_USERS:
        raise ValueError(f"Invalid user_id")
    
    # Predict for all items
    items = np.arange(N_ITEMS)
    users = np.full(N_ITEMS, user_id)
    
    scores = model.predict([users, items], verbose=0).flatten()
    
    # Get top K
    top_idx = scores.argsort()[-k:][::-1]
    
    return [
        {"item_id": int(items[i]), "score": float(scores[i])}
        for i in top_idx
    ]


if __name__ == "__main__":
    # Test
    recs = recommend(5, k=10)
    print(f"\nTop 10 recommendations:")
    for i, rec in enumerate(recs, 1):
        print(f"  {i}. Item {rec['item_id']} - {rec['score']:.4f}")
