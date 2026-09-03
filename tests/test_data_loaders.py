from collections import Counter
from collections.abc import Sized

from ml.data_loaders import create_data_loaders


def get_dataset_size(loader):
    dataset = loader.dataset

    if not isinstance(dataset, Sized):
        raise TypeError("Expected a dataset with a defined length.")

    return len(dataset)

train_loader, validation_loader, test_loader = create_data_loaders()

print(f"Training samples: {get_dataset_size(train_loader)}")
print(f"Validation samples: {get_dataset_size(validation_loader)}")
print(f"Test samples:       {get_dataset_size(test_loader)}")

print()

print(f"Training batches:   {len(train_loader)}")
print(f"Validation batches: {len(validation_loader)}")
print(f"Test batches:       {len(test_loader)}")

print()


def count_labels(loader):
    labels = []

    for _, batch_labels in loader:
        labels.extend(batch_labels.tolist())

    return Counter(labels)


print("Training labels:")
print(count_labels(train_loader))

print()

print("Validation labels:")
print(count_labels(validation_loader))

print()

print("Test labels:")
print(count_labels(test_loader))

print()

spectrograms, labels = next(iter(train_loader))

print("One training batch:")
print(f"Spectrogram shape: {spectrograms.shape}")
# shape understanding:
# The above prints: Spectrogram shape: torch.Size([4, 1, 64, 121])

# [4, 1, 64, 121]
#  │  │   │    │
#  │  │   │    └── time
#  │  │   └─────── Mel frequency bands
#  │  └─────────── audio/image channels
#  └────────────── batch size

# The CNN receives four at once

print(f"Labels: {labels}")