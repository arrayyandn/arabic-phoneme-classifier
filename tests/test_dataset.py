from ml.dataset import ArabicLetterDataset
from ml.labels import CLASSES

dataset = ArabicLetterDataset()

print(f"Number of samples: {len(dataset)}")
print()

for index in range(len(dataset)):
    mel, label = dataset[index]

    print(
        f"{index:02}: ", f"shape={tuple(mel.shape)}, {label=}, clases={CLASSES[label]}"
    )
