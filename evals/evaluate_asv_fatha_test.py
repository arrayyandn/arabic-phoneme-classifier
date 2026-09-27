import warnings

import torch
from scipy.io.wavfile import WavFileWarning
from sklearn.metrics import confusion_matrix
from torch import nn

from ml.asv_fatha_data_loaders import (
    create_asv_fatha_data_loaders,
)
from ml.fatha_labels import (
    FATHA_ARABIC_LABELS,
    FATHA_CLASSES,
)
from ml.model import ArabicLetterCNN

MODEL_FILE = (
    "models/checkpoints/"
    "asv_fatha_28_baseline.pt"
)


def main():
    # Some ASV WAV files contain metadata chunks that
    # scipy does not understand.
    
    # The actual audio is still read correctly.
    warnings.filterwarnings(
        "ignore",
        category=WavFileWarning,
    )

    _, _, test_loader = (
        create_asv_fatha_data_loaders(
            batch_size=16,
            augment_training=True,
        )
    )

    model = ArabicLetterCNN(
        num_classes=len(FATHA_CLASSES),
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

    test_loss = 0.0
    test_correct = 0
    test_samples = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for spectrograms, labels in test_loader:
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

            test_loss += (
                loss.item()
                * labels.size(0)
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

    test_accuracy = (
        test_correct
        / test_samples
    )

    print("ASV 28-class fatḥah test evaluation")
    print("-----------------------------------")
    print(f"Samples:  {test_samples}")
    print(f"Correct:  {test_correct}")
    print(f"Loss:     {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.1%}")

    print()
    print("Per-class accuracy")
    print("------------------")

    for class_index, class_name in enumerate(
        FATHA_CLASSES
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
            f"{class_index:>2} "
            f"{FATHA_ARABIC_LABELS[class_name]} "
            f"({class_name:<13}) "
            f"{class_correct:>2}/{class_total:<2} "
            f"{class_correct / class_total:6.1%}"
        )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(
            range(len(FATHA_CLASSES))
        ),
    )

    print()
    print("Confusion Matrix")
    print("----------------")
    print("Rows = actual")
    print("Columns = predicted")
    print()

    print(
        "    "
        + " ".join(
            f"{index:>2}"
            for index in range(
                len(FATHA_CLASSES)
            )
        )
    )

    for index, row in enumerate(matrix):
        values = " ".join(
            f"{value:>2}"
            for value in row
        )

        print(
            f"{index:>2}  {values}"
        )

    print()
    print("Class mapping")
    print("-------------")

    for index, class_name in enumerate(
        FATHA_CLASSES
    ):
        print(
            f"{index:>2} = "
            f"{FATHA_ARABIC_LABELS[class_name]} "
            f"({class_name})"
        )


if __name__ == "__main__":
    main()