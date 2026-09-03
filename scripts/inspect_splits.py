import random

from ml.data_loaders import (
    RANDOM_SEED,
    TRAIN_PER_CLASS,
    VALIDATION_PER_CLASS,
)
from dataset import (
    CLASSES,
    ArabicLetterDataset,
)

ARABIC = {
    "qaf": "قَ",
    "kaf": "كَ",
    "ta": "تَ",
    "taa_emphatic": "طَ",
    "sin": "سَ",
    "sad": "صَ",
}


def main():
    dataset = ArabicLetterDataset(augment=False)

    rng = random.Random(RANDOM_SEED)

    for class_index, class_name in enumerate(CLASSES):
        # Find all recordings belonging to this class.
        class_indices = [
            index
            for index, (_, label) in enumerate(dataset.samples)
            if label == class_index
        ]

        # Perform exactly the same deterministic shuffle
        # used by data_loaders.py.
        rng.shuffle(class_indices)

        # Split the shuffled indices using exactly the same
        # boundaries as our DataLoader code.
        train_indices = class_indices[:TRAIN_PER_CLASS]

        validation_indices = class_indices[
            TRAIN_PER_CLASS : TRAIN_PER_CLASS + VALIDATION_PER_CLASS
        ]

        test_indices = class_indices[TRAIN_PER_CLASS + VALIDATION_PER_CLASS :]

        print()
        print("=" * 50)
        print(f"{ARABIC[class_name]} ({class_name})")
        print("=" * 50)

        print()
        print("TRAINING:")

        for index in train_indices:
            audio_file, _ = dataset.samples[index]

            print(f"  {audio_file}")

        print()
        print("VALIDATION:")

        for index in validation_indices:
            audio_file, _ = dataset.samples[index]

            print(f"  {audio_file}")

        print()
        print("TEST:")

        for index in test_indices:
            audio_file, _ = dataset.samples[index]

            print(f"  {audio_file}")


if __name__ == "__main__":
    main()
