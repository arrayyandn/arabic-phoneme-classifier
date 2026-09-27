from pathlib import Path

from torch.utils.data import Dataset

from ml.audio_processing import (
    augment_waveform,
    load_wav,
    trim_and_center_waveform,
    waveform_to_mel,
)
from ml.labels import CLASSES

ASV_ROOT = Path(
    "data/external/asv/raw/alphabet_new_dataset"
)


# ASV contains 84 classes:

# 28 consonants
# ×
# 3 short vowels

# For this experiment we only use the six fatḥah
# classes already known by our current CNN.
ASV_FOLDER_BY_CLASS = {
    "qaf": "61",
    "kaf": "64",
    "ta": "07",
    "taa_emphatic": "46",
    "sin": "34",
    "sad": "40",
}


class ASVSixClassDataset(Dataset):
    def __init__(
        self,
        augment: bool = False,
    ):
        self.samples = []
        self.augment = augment

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

        waveform = load_wav(audio_file)

        waveform = trim_and_center_waveform(
            waveform
        )

        if self.augment:
            waveform = augment_waveform(
                waveform
            )

        mel = waveform_to_mel(
            waveform
        )

        return mel, label