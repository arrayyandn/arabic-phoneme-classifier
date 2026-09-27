import warnings
from pathlib import Path

import torch
from scipy.io.wavfile import WavFileWarning
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

from ml.audio_processing import (
    load_wav,
    trim_and_center_waveform,
    waveform_to_mel,
)
from ml.labels import ARABIC_LABELS, CLASSES
from ml.model import ArabicLetterCNN


MODEL_FILE = Path(
    "models/checkpoints/six_class_dropout_030.pt"
)

ASV_ROOT = Path(
    "data/external/asv/raw/alphabet_new_dataset"
)


# The ASV dataset contains:

# 28 Arabic letters
# ×
# 3 short vowels
# =
# 84 classes

# These are the ASV folders corresponding to the
# six fatḥah classes our existing model already knows.
ASV_FOLDER_BY_CLASS = {
    "qaf": "61",
    "kaf": "64",
    "ta": "07",
    "taa_emphatic": "46",
    "sin": "34",
    "sad": "40",
}


class ASVSixClassDataset(Dataset):
    def __init__(self):
        self.samples = []

        for class_index, class_name in enumerate(CLASSES):
            class_folder = (
                ASV_ROOT
                / ASV_FOLDER_BY_CLASS[class_name]
            )

            audio_files = sorted(
                class_folder.glob("*.wav")
            )

            if not audio_files:
                raise FileNotFoundError(
                    f"No WAV files found in "
                    f"{class_folder}"
                )

            for audio_file in audio_files:
                self.samples.append(
                    (
                        audio_file,
                        class_index,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio_file, label = self.samples[index]

        # Use exactly the same preprocessing pipeline
        # that our existing model was trained with.
        waveform = load_wav(audio_file)

        waveform = trim_and_center_waveform(
            waveform
        )

        mel = waveform_to_mel(waveform)

        return mel, label


def main():
    # Some ASV WAV files contain extra metadata chunks
    # which scipy does not understand.

    # scipy still reads the actual audio correctly, so
    # suppress that warning while evaluating hundreds
    # of recordings.
    warnings.filterwarnings(
        "ignore",
        category=WavFileWarning,
    )

    dataset = ASVSixClassDataset()

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )

    print("ASV six-class external evaluation")
    print("---------------------------------")
    print(f"Samples: {len(dataset)}")
    print()

    print("Samples per class:")

    for class_index, class_name in enumerate(CLASSES):
        count = sum(
            label == class_index
            for _, label in dataset.samples
        )

        print(
            f"{ARABIC_LABELS[class_name]} "
            f"({class_name:<12}) "
            f"{count}"
        )

    model = ArabicLetterCNN(
        num_classes=len(CLASSES)
    )

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

    with torch.no_grad():
        for spectrograms, labels in loader:
            outputs = model(spectrograms)

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.tolist()
            )

            all_predictions.extend(
                predictions.tolist()
            )

    correct = sum(
        actual == predicted
        for actual, predicted in zip(
            all_labels,
            all_predictions,
        )
    )

    total = len(all_labels)

    accuracy = correct / total

    print()
    print("Overall results")
    print("---------------")
    print(f"Correct:  {correct}/{total}")
    print(f"Accuracy: {accuracy:.1%}")

    print()
    print("Per-class accuracy")
    print("------------------")

    for class_index, class_name in enumerate(CLASSES):
        class_results = [
            actual == predicted
            for actual, predicted in zip(
                all_labels,
                all_predictions,
            )
            if actual == class_index
        ]

        class_correct = sum(class_results)
        class_total = len(class_results)

        print(
            f"{ARABIC_LABELS[class_name]} "
            f"({class_name:<12}) "
            f"{class_correct:>3}/{class_total:<3} "
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
    print("Actual \\ Predicted")

    print(
        "                  "
        + " ".join(
            f"{index:>3}"
            for index in range(len(CLASSES))
        )
    )

    for index, row in enumerate(matrix):
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

    for index, class_name in enumerate(CLASSES):
        print(
            f"{index} = "
            f"{ARABIC_LABELS[class_name]} "
            f"({class_name})"
        )


if __name__ == "__main__":
    main()