from pathlib import Path


# Reproducibility
SEED = 42


# Image configuration
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)


# Training configuration
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001
DROPOUT_RATE = 0.3


# Directory paths
DEEP_LEARNING_DIR = Path(__file__).resolve().parents[2]

DATASET_DIR = DEEP_LEARNING_DIR / "datasets" / "aircraft_damage_dataset_v1"

TRAIN_DIR = DATASET_DIR / "train"
VALID_DIR = DATASET_DIR / "valid"
TEST_DIR = DATASET_DIR / "test"

PROJECT_DIR = DEEP_LEARNING_DIR / "15-Aircraft-Damage-Classification"
OUTPUTS_DIR = PROJECT_DIR / "OUTPUTS"