from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam

from .config import IMAGE_SIZE


def build_model():

    # Load pretrained VGG16 without its original classifier
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )

    # Freeze pretrained VGG16 layers
    base_model.trainable = False

    # Build a smaller classifier
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])

    # Configure training
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model