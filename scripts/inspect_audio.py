from pathlib import Path

import matplotlib.pyplot as plt

from audio_processing import (
    SAMPLE_RATE,
    load_wav,
    trim_and_center_waveform,
    waveform_to_mel,
)

AUDIO_FILES = [
    Path(r"data\qaf\qaf_005.wav"),
    Path(r"data\taa_emphatic\taa_emphatic_013.wav"),
    Path(r"data\taa_emphatic\taa_emphatic_005.wav"),
    Path(r"data\kaf\kaf_012.wav"),
    Path(r"data\kaf\kaf_004.wav"),
    Path(r"data\sad\sad_013.wav"),
    Path(r"data\sad\sad_003.wav"),
]


def create_time_axis(waveform):
    # Convert:
    # sample 0
    # sample 1
    # sample 2
    # ...
    #
    # into:
    # seconds

    return [sample / SAMPLE_RATE for sample in range(waveform.shape[1])]


def show_waveform(
    waveform,
    audio_file,
    title_prefix,
):
    time_axis = create_time_axis(waveform)

    plt.figure(figsize=(12, 4))

    plt.plot(
        time_axis,
        waveform[0],
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.title(f"{title_prefix} — {audio_file}")

    plt.tight_layout()
    plt.show()


def show_mel(
    waveform,
    audio_file,
):
    # Convert the CENTRED waveform into exactly the same
    # Mel representation that the CNN receives.
    mel = waveform_to_mel(waveform)

    # mel has shape:
    # [1, 64, 121]
    #
    # 1   = audio channel
    # 64  = Mel frequency bands
    # 121 = time positions

    plt.figure(figsize=(12, 5))

    plt.imshow(
        mel[0].numpy(),
        origin="lower",
        aspect="auto",
    )

    plt.xlabel("Time frame")
    plt.ylabel("Mel frequency band")

    plt.title(f"Centred Normalised Mel Spectrogram — {audio_file}")

    plt.colorbar(label="Normalised intensity")

    plt.tight_layout()
    plt.show()


def inspect_audio(audio_file: Path):
    # Load the ORIGINAL WAV file.
    original_waveform = load_wav(audio_file)

    # Apply the same deterministic centring step used by:
    # training
    # validation
    # test
    # live prediction
    centred_waveform = trim_and_center_waveform(original_waveform)

    # 1. ORIGINAL WAVEFORM

    # Shows where you actually happened to pronounce
    # the sound during the 1.2-second recording.
    show_waveform(
        original_waveform,
        audio_file,
        "Original waveform",
    )

    # 2. CENTRED WAVEFORM

    # Shows the waveform AFTER excessive surrounding
    # silence has been removed and the pronunciation
    # has been placed in the centre.
    #
    # The sound itself is NOT stretched or shortened.
    show_waveform(
        centred_waveform,
        audio_file,
        "Centred waveform",
    )

    # 3. CENTRED MEL SPECTROGRAM

    # This is now representative of what the CNN
    # actually receives.
    show_mel(
        centred_waveform,
        audio_file,
    )


def main():
    for audio_file in AUDIO_FILES:
        print()
        print("=" * 60)
        print(f"Inspecting: {audio_file}")
        print("=" * 60)

        inspect_audio(audio_file)


if __name__ == "__main__":
    main()
