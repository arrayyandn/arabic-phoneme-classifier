import torch

from ml.dataset import ArabicLetterDataset

training_dataset = ArabicLetterDataset(augment=True)

normal_dataset = ArabicLetterDataset(augment=False)


# Fetch index 0 twice from the augmented dataset.
augmented_1, label_1 = training_dataset[0]
augmented_2, label_2 = training_dataset[0]


# Fetch index 0 twice without augmentation.
normal_1, label_3 = normal_dataset[0]
normal_2, label_4 = normal_dataset[0]


print("Augmentation test")
print("-----------------")
print()

print(f"Labels: {label_1}, {label_2}, {label_3}, {label_4}")

print()

print(
    "Two augmented versions identical:",
    torch.equal(
        augmented_1,
        augmented_2,
    ),
)

print(
    "Two normal versions identical:",
    torch.equal(
        normal_1,
        normal_2,
    ),
)

print()

print(
    "Mean absolute difference between augmented versions:",
    (augmented_1 - augmented_2).abs().mean().item(),
)

print(
    "Mean absolute difference between normal versions:",
    (normal_1 - normal_2).abs().mean().item(),
)
