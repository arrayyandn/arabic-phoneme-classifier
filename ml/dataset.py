from pathlib import Path

from audio_processing import (
    augment_waveform,
    load_wav,
    trim_and_center_waveform,
    waveform_to_mel,
)
from torch.utils.data import Dataset

DATA_DIR = Path("data/original")

CLASSES = [
    "qaf",
    "kaf",
    "ta",
    "taa_emphatic",
    "sin",
    "sad",
]


class ArabicLetterDataset(Dataset):
    def __init__(self, dat_dir: Path = DATA_DIR, augment: bool = False):
        self.samples = []

        # augment determines whether random data augmentation
        # should be applied when a recording is loaded.
        
        # Training dataset:
            # augment=True
        
        # Validation/test datasets:
            # augment=False
        
        # We NEVER augment validation or test data because those
        # datasets are supposed to represent genuine unseen recordings.

        self.augment = augment
        
        for class_index, class_name in enumerate(CLASSES):
            class_dir = dat_dir / class_name

            for audio_file in sorted(class_dir.glob("*.wav")):
                self.samples.append((audio_file, class_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        audio_file, label = self.samples[index]

        # Load the WAV file and convert it into a PyTorch waveform:
        
        # WAV file
        #    ↓
        # NumPy array
        #    ↓
        # PyTorch Tensor
        #    ↓
        # [channels, samples]

        waveform = load_wav(audio_file)

        # Remove accidental variation in WHERE the speaker
        # happened to begin speaking inside the 1.2-second window.
        
        # This is deterministic and therefore happens for:
            # training
            # validation
            # test
            # prediction
        waveform = trim_and_center_waveform(
            waveform
        )

        # DATA AUGMENTATION:
            # Only apply this when this particular Dataset
            # was created with:
                # augment=True

            # Training:
                # augment=True
                #      ↓
                # same original WAV file
                #      ↓
                # slightly different waveform each time it is loaded

            # Validation / Test:
                # augment=False
                #      ↓
                # original waveform is left unchanged

            # This is important because we want augmentation to make
            # the training data harder to memorise, while validation
            # and test data must remain genuine unchanged recordings.
        
        # Only training receives random augmentation.
        
        # The sound has already been centred, then augmentation
        # deliberately introduces a small controlled amount
        # of timing variation again.

        if self.augment:
            waveform = augment_waveform(waveform)

        # Convert the waveform into the representation the CNN receives:
        
        # waveform
        #    ↓
        # Mel spectrogram
        #    ↓
        # decibel scale
        #    ↓
        # normalisation
        #    ↓
        # approximately [1, 64, 121]

        mel_db = waveform_to_mel(waveform)

        return mel_db, label