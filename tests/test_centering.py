from pathlib import Path

from ml.audio_processing import (
    TARGET_NUM_SAMPLES,
    load_wav,
    trim_and_center_waveform,
)

AUDIO_FILES = [
    Path(r"data/original/qaf/qaf_005.wav"),
    Path(r"data/original/kaf/kaf_012.wav"),
    Path(r"data/original/kaf/kaf_004.wav"),
    Path(r"data/original/taa_emphatic/taa_emphatic_005.wav"),
    Path(r"data/original/sad/sad_003.wav"),
]


print("Centering test")
print("--------------")
print()


for audio_file in AUDIO_FILES:
    original = load_wav(audio_file)

    centred = trim_and_center_waveform(
        original
    )

    print(audio_file)

    print(
        f"  Original shape: {tuple(original.shape)}"
    )

    print(
        f"  Centred shape:  {tuple(centred.shape)}"
    )

    print(
        f"  Correct length: "
        f"{centred.shape[1] == TARGET_NUM_SAMPLES}"
    )

    print()