import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from ml.data_loaders import create_data_loaders
from ml.dataset import CLASSES
from ml.model import ArabicLetterCNN

MODEL_FILE = "models/latest/arabic_letter_cnn.pt"

ARABIC = {
    "qaf": "قَ",
    "kaf": "كَ",
    "ta": "تَ",
    "taa_emphatic": "طَ",
    "sin": "سَ",
    "sad": "صَ",
}


def main():
    # We deliberately ignore the training and validation loaders here.
    _, _, test_loader = create_data_loaders(
        batch_size=4
    )

    model = ArabicLetterCNN(num_classes=len(CLASSES))

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    loss_function = nn.CrossEntropyLoss()

    test_loss = 0.0
    test_correct = 0
    test_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for spectrograms, labels in test_loader:

            # Forward pass only.
            # No training or weights updates happen here.

            outputs = model(spectrograms)

            loss = loss_function(
                outputs, 
                labels
            )

            predictions = outputs.argmax(dim=1)

            test_loss += (
                loss.item() * labels.size(0)
            )

            test_correct += (
                predictions == labels
            ).sum().item()

            test_samples += labels.size(0)

            all_labels.extend(
                labels.tolist()
            )

            all_predictions.extend(
                predictions.tolist()
            )

    test_loss /= test_samples
    test_accuracy = test_correct / test_samples

    print("Test Results")
    print("------------")
    print(f"Test samples:  {test_samples}")
    print(f"Test loss:     {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.1%}")

    print()
    print("Individual predictions")
    print("----------------------")

    for actual, predicted in zip(
        all_labels,
        all_predictions
    ):
        actual_name = CLASSES[actual]
        predicted_name = CLASSES[predicted]

        result = (
            "✓"
            if actual == predicted
            else "✗"
        )

        print(
            f"{result} "
            f"Actual: {ARABIC[actual_name]} "
            f"({actual_name:<12}) "
            f"Predicted: {ARABIC[predicted_name]} "
            f"({predicted_name})"
        )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASSES)))
    )

    print()
    print("Confusion Matrix")
    print("----------------")
    print(
        "Actual \\ Predicted"
    )
    print(
        "          "
        + " ".join(
            f"{i:>3}"
            for i in range(len(CLASSES))
        )
    )
    for index, row in enumerate(matrix):
        values = " ".join(
            f"{value:>3}"
            for value in row
        )

        print(
            f"{index} {CLASSES[index]:<12} "
            f"{values}"
        )

    print()
    print("Class mapping:")

    for index, class_name in enumerate(CLASSES):
        print(
            f"{index} = "
            f"{ARABIC[class_name]} "
            f"({class_name})"
        )


if __name__ == "__main__":
    main()


