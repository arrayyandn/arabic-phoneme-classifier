from pathlib import Path

from torch.utils.data import Dataset

from ml.audio_processing import (
    augment_waveform,
    load_wav,
    trim_and_center_waveform,
    waveform_to_mel,
)
from ml.fatha_labels import FATHA_CLASSES

ASV_ROOT = Path(
    "data/external/asv/raw/alphabet_new_dataset"
)


# ASV contains 84 classes:

# 28 Arabic consonants
# ×
# 3 short vowels

# The fatḥah classes are every third folder,
# beginning with folder 01.
ASV_FATHA_FOLDER_BY_CLASS = {
    "hamza": "01",
    "ba": "04",
    "ta": "07",
    "tha": "10",
    "jim": "13",
    "haa": "16",
    "kha": "19",
    "dal": "22",
    "dhal": "25",
    "ra": "28",
    "zay": "31",
    "sin": "34",
    "shin": "37",
    "sad": "40",
    "dad": "43",
    "taa_emphatic": "46",
    "zaa_emphatic": "49",
    "ayn": "52",
    "ghayn": "55",
    "fa": "58",
    "qaf": "61",
    "kaf": "64",
    "lam": "67",
    "mim": "70",
    "nun": "73",
    "ha": "76",
    "waw": "79",
    "ya": "82",
}


class ASVFathaDataset(Dataset):
    def __init__(
        self,
        augment: bool = False,
    ):
        self.samples = []
        self.augment = augment

        for class_index, class_name in enumerate(
            FATHA_CLASSES
        ):
            class_folder = (
                ASV_ROOT
                / ASV_FATHA_FOLDER_BY_CLASS[
                    class_name
                ]
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

        waveform = load_wav(
            audio_file
        )

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