from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

from .config import (
    TRAIN_DIR,
    VALID_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
)

CLASS_NAMES = ["crack", "dent"]


def create_data_generators():

    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = data_generator.flow_from_directory(
        directory=str(TRAIN_DIR),
        classes=CLASS_NAMES,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=SEED,
    )

    valid_generator = data_generator.flow_from_directory(
        directory=str(VALID_DIR),
        classes=CLASS_NAMES,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    test_generator = data_generator.flow_from_directory(
        directory=str(TEST_DIR),
        classes=CLASS_NAMES,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    return train_generator, valid_generator, test_generator