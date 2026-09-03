import time
from pathlib import Path

import sounddevice as sd
import torch
from scipy.io.wavfile import write

from ml.audio_processing import (
    SAMPLE_RATE,
    TARGET_DURATION,
    load_wav,
    trim_and_center_waveform,
    waveform_to_mel,
)
from ml.dataset import CLASSES
from ml.model import ArabicLetterCNN

MODEL_FILE = Path("models/latest/arabic_letter_cnn.pt")
RECORDING_FILE = Path("outputs/audio/prediction.wav")

DURATION = TARGET_DURATION


ARABIC = {
    "qaf": "قَ",
    "kaf": "كَ",
    "ta": "تَ",
    "taa_emphatic": "طَ",
    "sin": "سَ",
    "sad": "صَ",
}


def record_audio():
    input("Press Enter when ready...")

    print("3...")
    time.sleep(0.2)

    print("2...")
    time.sleep(0.2)

    print("1...")
    time.sleep(0.2)

    print("Recording...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    RECORDING_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write(
        RECORDING_FILE,
        SAMPLE_RATE,
        audio,
    )

    print("Recording finished.")


def main():
    # Load trained neural network

    model = ArabicLetterCNN(num_classes=len(CLASSES))

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()

    # Record new unseen audio

    print()
    print("Arabic Letter Classifier")
    print("------------------------")
    print()
    print("Pronounce one of:")
    print("قَ  كَ  تَ  طَ  سَ  صَ")
    print()

    record_audio()

    # Preprocess audio

    waveform = load_wav(RECORDING_FILE)

    waveform = trim_and_center_waveform(
        waveform
    )

    mel = waveform_to_mel(waveform)

    # mel currently has:
    #
    # [1, 64, 121]
    #
    # But the CNN expects:
    #
    # [batch, channel, frequency, time]
    #
    # So add a batch dimension:
    #
    # [1, 64, 121]
    #       ↓
    # [1, 1, 64, 121]

    mel = mel.unsqueeze(0)

    # Make prediction

    with torch.no_grad():
        logits = model(mel)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    probabilities = probabilities[0]

    predicted_index: int = int(probabilities.argmax().item())
    predicted_class = CLASSES[predicted_index]

    # Display results

    print()
    print("Prediction probabilities:")
    print("-------------------------")

    for index, class_name in enumerate(CLASSES):
        probability = probabilities[index].item()

        print(
            f"{ARABIC[class_name]} "
            f"({class_name:<12}) "
            f"{probability:6.2%}"
        )

    print()
    print(
        f"Prediction: "
        f"{ARABIC[predicted_class]} "
        f"({predicted_class})"
    )


if __name__ == "__main__":
    main()