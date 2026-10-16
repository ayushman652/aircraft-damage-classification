from src.preprocessing import create_data_generators
from src.model import build_model
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.visualizer import plot_training_history


def main():

    # 1. Prepare the datasets
    train_generator, valid_generator, test_generator = ( create_data_generators() )

    # 2. Build the VGG16 model
    model = build_model()
    
    # 3. Display the model architecture
    model.summary()

    # 4. Train the model
    history = train_model(
        model,
        train_generator,
        valid_generator,
    )

    # 5. Plot and save training history
    plot_training_history(history)

    # 6. Evaluate on the test dataset
    evaluate_model(model, test_generator)


if __name__ == "__main__":
    main()