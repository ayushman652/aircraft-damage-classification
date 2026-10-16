import matplotlib.pyplot as plt

from .config import OUTPUTS_DIR


def plot_training_history(history):

    # Create outputs directory if it doesn't exist
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = history.history
    epochs = range(1, len(metrics["loss"]) + 1)

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics["accuracy"], label="Training Accuracy")
    plt.plot(epochs, metrics["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUTS_DIR / "training_accuracy.png", dpi=300)
    plt.close()

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metrics["loss"], label="Training Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUTS_DIR / "training_loss.png", dpi=300)
    plt.close()

    print(f"Training plots saved to: {OUTPUTS_DIR}")