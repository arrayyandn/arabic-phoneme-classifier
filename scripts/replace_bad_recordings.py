from pathlib import Path

from scipy.io.wavfile import write

from scripts.record_dataset import (
    SAMPLE_RATE,
    choose_recording_actions,
    play_audio,
    record_audio,
)

REPLACEMENTS = [
    (
        Path("data/original/kaf/kaf_004.wav"),
        "كَ",
    ),
    (
        Path("data/original/taa_emphatic/taa_emphatic_014.wav"),
        "طَ",
    ),
]


def main():
    print("Replace Bad Dataset Recordings")
    print("------------------------------")

    for audio_file, arabic_sound in REPLACEMENTS:
        print()
        print("=" * 40)
        print(f"Replacing: {audio_file}")
        print(f"Pronounce: {arabic_sound}")
        print("=" * 40)

        input("Press Enter when ready...")

        audio = record_audio()

        while True:
            action = choose_recording_actions()

            if action == "p":
                play_audio(audio)

            elif action == "r":
                print()
                print("Retaking...")

                input("Press Enter when ready...")

                audio = record_audio()

            elif action == "a":
                # Overwrite the existing bad recording
                # while keeping exactly the same filename.
                
                # This means our deterministic train /
                # validation / test split does not change.
                write(
                    audio_file,
                    SAMPLE_RATE,
                    audio,
                )

                print(f"Replaced: {audio_file}")

                break

            elif action == "q":
                print("Stopped.")
                return

    print()
    print("Both recordings replaced.")


if __name__ == "__main__":
    main()
