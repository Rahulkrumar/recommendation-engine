"""
Training pipeline for recommendation engine
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model import build_ncf
from config import *
import os

np.random.seed(42)


def generate_data():
    """Generate synthetic user-item interactions"""
    print("Generating interaction data...")
    
    users = np.random.randint(0, N_USERS, N_INTERACTIONS)
    items = np.random.randint(0, N_ITEMS, N_INTERACTIONS)
    ratings = np.random.choice([0, 1], p=[0.7, 0.3], size=N_INTERACTIONS)
    
    df = pd.DataFrame({
        "user_id": users, 
        "item_id": items, 
        "rating": ratings
    })
    
    print(f"Generated {len(df):,} interactions")
    return df


def main():
    """Main training function"""
    
    print("="*60)
    print("RECOMMENDATION ENGINE - TRAINING")
    print("="*60)
    
    # Generate data
    df = generate_data()
    
    # Prepare data
    X = df[["user_id", "item_id"]].values
    y = df["rating"].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Test: {len(X_test):,} samples")
    
    # Build model
    print("\nBuilding model...")
    model = build_ncf(N_USERS, N_ITEMS, EMBEDDING_DIM)
    
    # Train
    print("\nTraining...")
    model.fit(
        [X_train[:,0], X_train[:,1]],
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating...")
    test_loss, test_acc = model.evaluate(
        [X_test[:,0], X_test[:,1]],
        y_test,
        verbose=0
    )
    
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    
    print(f"\n✅ Model saved to {MODEL_PATH}")
    print("="*60)


if __name__ == "__main__":
    main()
