import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import Subset

from ml.data_loaders import create_data_loaders
from ml.dataset import CLASSES, ArabicLetterDataset
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
    # We deliberately use the VALIDATION set here.
    
    # Training set:
    # used to learn weights
    
    # Validation set:
    # allowed to guide model development
    
    # Legacy test set:
        # already inspected during development
        # so it is now only useful for historical comparison

    _, validation_loader, _ = create_data_loaders(
        batch_size=4,
        augment_training=True,
    )

    # FIND THE EXACT VALIDATION FILES
    
    # validation_loader.dataset is a Subset.
    
    # The Subset contains:
        # dataset:
            # the original ArabicLetterDataset
        #
        # indices:
            # which recordings from that dataset belong
            # to the validation split

    validation_subset = validation_loader.dataset

    if not isinstance(validation_subset, Subset):
        raise TypeError(
            "Expected validation dataset to be a Subset."
        )

    source_dataset = validation_subset.dataset

    if not isinstance(source_dataset, ArabicLetterDataset):
        raise TypeError(
            "Expected an ArabicLetterDataset."
        )

    validation_files = [
        source_dataset.samples[index][0]
        for index in validation_subset.indices
    ]

    model = ArabicLetterCNN(num_classes=len(CLASSES))

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for spectrograms, labels in validation_loader:
            # Ask the saved model for its six logits.
            outputs = model(spectrograms)

            # Convert those logits into probabilities which
            # sum to 100% for each recording.
            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            # argmax finds the class with the largest logit.
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.tolist())

            all_predictions.extend(predictions.tolist())

            all_probabilities.extend(
                probabilities.tolist()
            )

    correct = sum(
        actual == predicted
        for actual, predicted in zip(
            all_labels,
            all_predictions,
        )
    )

    total = len(all_labels)

    print("Validation Results")
    print("------------------")
    print(f"Samples:  {total}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {correct / total:.1%}")

    print()
    print("Individual predictions")
    print("----------------------")

    for (
        audio_file,
        actual,
        predicted,
        probabilities,
    ) in zip(
        validation_files,
        all_labels,
        all_predictions,
        all_probabilities,
    ):

        actual_name = CLASSES[actual]
        predicted_name = CLASSES[predicted]

        result = (
            "✓"
            if actual == predicted
            else "✗"
        )

        print()
        print(
            f"{result} {audio_file}"
        )

        print(
            f"Actual:    "
            f"{ARABIC[actual_name]} "
            f"({actual_name})"
        )

        print(
            f"Predicted: "
            f"{ARABIC[predicted_name]} "
            f"({predicted_name})"
        )

        print("Probabilities:")

        for class_index, class_name in enumerate(CLASSES):
            print(
                f"    "
                f"{ARABIC[class_name]} "
                f"{class_name:<12} "
                f"{probabilities[class_index]:6.2%}"
            )

    # CONFUSION MATRIX:
    
    # Rows:
    # actual classes
    
    # Columns:
    # predicted classes
    
    # Example:
    
    #              predicted
    #             تَ    طَ
    # actual تَ    2     1
    #        طَ    1     2
    
    # would tell us that the model sometimes confuses
    # تَ and طَ with each other.

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(CLASSES))),
    )

    print()
    print("Confusion Matrix")
    print("----------------")
    print("Actual \\ Predicted")

    print(
        "                  " + " ".join(f"{index:>3}" for index in range(len(CLASSES)))
    )

    for index, row in enumerate(matrix):
        values = " ".join(f"{value:>3}" for value in row)

        print(f"{index} {CLASSES[index]:<12} {values}")

    print()
    print("Class mapping:")

    for index, class_name in enumerate(CLASSES):
        print(f"{index} = {ARABIC[class_name]} ({class_name})")


if __name__ == "__main__":
    main()
