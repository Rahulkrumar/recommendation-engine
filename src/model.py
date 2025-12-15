"""
Neural Collaborative Filtering model architecture
"""

from tensorflow import keras


def build_ncf(n_users, n_items, embedding_dim=64):
    """
    Build Neural Collaborative Filtering model
    
    Args:
        n_users: Number of users
        n_items: Number of items
        embedding_dim: Dimension of embeddings
    
    Returns:
        Compiled Keras model
    """
    
    # User tower
    user_input = keras.layers.Input(shape=(1,), name='user_input')
    user_emb = keras.layers.Embedding(
        n_users, 
        embedding_dim,
        name='user_embedding'
    )(user_input)
    user_vec = keras.layers.Flatten()(user_emb)

    # Item tower
    item_input = keras.layers.Input(shape=(1,), name='item_input')
    item_emb = keras.layers.Embedding(
        n_items, 
        embedding_dim,
        name='item_embedding'
    )(item_input)
    item_vec = keras.layers.Flatten()(item_emb)

    # Interaction layers
    x = keras.layers.Concatenate()([user_vec, item_vec])
    
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    
    x = keras.layers.Dense(64, activation="relu")(x)

    # Output
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    # Build and compile
    model = keras.Model(
        inputs=[user_input, item_input], 
        outputs=output,
        name='NCF'
    )
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model
