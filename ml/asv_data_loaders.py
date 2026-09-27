import random

from torch.utils.data import DataLoader, Subset

from ml.asv_dataset import ASVSixClassDataset
from ml.labels import CLASSES

RANDOM_SEED = 42

TRAIN_RATIO = 0.80


def create_asv_data_loaders(
    batch_size=8,
    augment_training=True,
):
    split_dataset = ASVSixClassDataset(
        augment=False
    )

    training_source = ASVSixClassDataset(
        augment=augment_training
    )

    validation_source = ASVSixClassDataset(
        augment=False
    )

    train_indices = []
    validation_indices = []

    rng = random.Random(
        RANDOM_SEED
    )

    for class_index in range(len(CLASSES)):
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

        train_indices.extend(
            class_indices[:train_count]
        )

        validation_indices.extend(
            class_indices[train_count:]
        )

    train_dataset = Subset(
        training_source,
        train_indices,
    )

    validation_dataset = Subset(
        validation_source,
        validation_indices,
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

    return (
        train_loader,
        validation_loader,
    )