import random

from torch.utils.data import DataLoader, Subset

from ml.asv_fatha_dataset import ASVFathaDataset
from ml.fatha_labels import FATHA_CLASSES

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def create_asv_fatha_data_loaders(
    batch_size=16,
    augment_training=True,
):
    # Dataset used only to determine which file indices
    # belong to each split.
    split_dataset = ASVFathaDataset(
        augment=False
    )

    # Only the training source receives augmentation.
    training_source = ASVFathaDataset(
        augment=augment_training
    )

    # Validation and test recordings must remain unchanged.
    evaluation_source = ASVFathaDataset(
        augment=False
    )

    train_indices = []
    validation_indices = []
    test_indices = []

    rng = random.Random(
        RANDOM_SEED
    )

    for class_index in range(
        len(FATHA_CLASSES)
    ):
        class_indices = [
            index
            for index, (_, label)
            in enumerate(split_dataset.samples)
            if label == class_index
        ]

        rng.shuffle(
            class_indices
        )

        train_count = int(
            len(class_indices)
            * TRAIN_RATIO
        )

        validation_count = int(
            len(class_indices)
            * VALIDATION_RATIO
        )

        train_end = train_count

        validation_end = (
            train_end
            + validation_count
        )

        train_indices.extend(
            class_indices[:train_end]
        )

        validation_indices.extend(
            class_indices[
                train_end:validation_end
            ]
        )

        test_indices.extend(
            class_indices[
                validation_end:
            ]
        )

    train_dataset = Subset(
        training_source,
        train_indices,
    )

    validation_dataset = Subset(
        evaluation_source,
        validation_indices,
    )

    test_dataset = Subset(
        evaluation_source,
        test_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )