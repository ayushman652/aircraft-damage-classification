def evaluate_model(model, test_generator):

    results = model.evaluate(
        test_generator,
        verbose=1,
    )

    test_loss = results[0]
    test_accuracy = results[1]

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2%}")

    return results