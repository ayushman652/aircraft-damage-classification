from tensorflow.keras.callbacks import EarlyStopping


def train_model(model, train_generator, valid_generator, epochs=30):

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        callbacks=[early_stopping],
    )

    return history