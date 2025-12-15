"""
Configuration for recommendation engine
"""

# Data configuration
N_USERS = 10000
N_ITEMS = 5000
N_INTERACTIONS = 2000000

# Model configuration
EMBEDDING_DIM = 64
HIDDEN_LAYERS = [256, 128, 64]
DROPOUT_RATE = 0.3

# Training configuration
BATCH_SIZE = 512
EPOCHS = 10
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.001

# Paths
MODEL_PATH = "models/ncf_model.keras"
DATA_PATH = "data/interactions.csv"
