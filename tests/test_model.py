from ml.data_loaders import create_data_loaders
from ml.model import ArabicLetterCNN

train_loader, _, _ = create_data_loaders()

model = ArabicLetterCNN()

spectrograms, labels = next(iter(train_loader))

outputs = model(spectrograms)

print(f"Input shape:  {spectrograms.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Output shape: {outputs.shape}")

print()
print("Raw model outputs:")
print(outputs)
