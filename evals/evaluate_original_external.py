import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from ml.dataset import ArabicLetterDataset
from ml.labels import ARABIC_LABELS, CLASSES
from ml.model import ArabicLetterCNN

MODEL_FILE = (
    "models/checkpoints/"
    "asv_six_class_baseline_500.pt"
)


def main():
    # This model was trained using ASV rather than
    # our original recordings.
    
    # Therefore all 90 original recordings can be
    # evaluated as a separate external corpus.
    dataset = ArabicLetterDataset(
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )

    model = ArabicLetterCNN(
        num_classes=len(CLASSES),
        dropout_p=0.0,
    )

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for spectrograms, labels in loader:
            outputs = model(
                spectrograms
            )

            loss = loss_function(
                outputs,
                labels,
            )

            predictions = outputs.argmax(
                dim=1
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

            all_labels.extend(
                labels.tolist()
            )

            all_predictions.extend(
                predictions.tolist()
            )

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        total_correct
        / total_samples
    )

    print(
        "Original corpus external evaluation"
    )
    print(
        "-----------------------------------"
    )
    print(
        f"Samples:  {total_samples}"
    )
    print(
        f"Correct:  {total_correct}"
    )
    print(
        f"Loss:     {average_loss:.4f}"
    )
    print(
        f"Accuracy: {accuracy:.1%}"
    )

    print()
    print("Per-class accuracy")
    print("------------------")

    for class_index, class_name in enumerate(
        CLASSES
    ):
        class_results = [
            actual == predicted
            for actual, predicted in zip(
                all_labels,
                all_predictions,
            )
            if actual == class_index
        ]

        class_correct = sum(
            class_results
        )

        class_total = len(
            class_results
        )

        print(
            f"{ARABIC_LABELS[class_name]} "
            f"({class_name:<12}) "
            f"{class_correct:>2}/{class_total:<2} "
            f"{class_correct / class_total:6.1%}"
        )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(
            range(len(CLASSES))
        ),
    )

    print()
    print("Confusion Matrix")
    print("----------------")
    print(
        "Actual \\ Predicted"
    )

    print(
        "                  "
        + " ".join(
            f"{index:>3}"
            for index in range(
                len(CLASSES)
            )
        )
    )

    for index, row in enumerate(
        matrix
    ):
        values = " ".join(
            f"{value:>3}"
            for value in row
        )

        print(
            f"{index} "
            f"{CLASSES[index]:<12} "
            f"{values}"
        )

    print()
    print("Class mapping:")

    for index, class_name in enumerate(
        CLASSES
    ):
        print(
            f"{index} = "
            f"{ARABIC_LABELS[class_name]} "
            f"({class_name})"
        )


if __name__ == "__main__":
    main()