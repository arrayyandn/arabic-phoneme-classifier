import time
from pathlib import Path

import sounddevice as sd
from scipy.io.wavfile import read, write

SAMPLE_RATE = 16_000
DURATION = 1.2
TARGET_RECORDINGS_PER_LETTER = 15
SLEEP_DURATION = 0.2

LETTERS = {
    "qaf": "قَ",
    "kaf": "كَ",
    "ta": "تَ",
    "taa_emphatic": "طَ",
    "sin": "سَ",
    "sad": "صَ",
}

DATA_DIR = Path("data/original")


def record_audio():
    """
    Record one audio sample and return it.

    We do NOT immediately save the recording.

    This lets us:
        - play it back
        - accept it
        - reject it
        - retake it

    without replacing a good existing WAV file.
    """
    print("3...")
    time.sleep(SLEEP_DURATION)

    print("2...")
    time.sleep(SLEEP_DURATION)

    print("1...")
    time.sleep(SLEEP_DURATION)

    print("Recording...")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32"
    )

    sd.wait()

    print("Recording finished.")

    return audio

def play_audio(audio):
    """
    Play a newly-recorded sample through the speakers/headphones.
    """

    print("Playing recording...")

    sd.play(
        audio,
        samplerate=SAMPLE_RATE
    )

    sd.wait()

    print("Playback finished.")

def get_next_recording_number(letter_dir: Path, folder_name: str):
    """
    Find the next recording number automatically.

    Example:

        qaf_001.wav
        qaf_002.wav
        qaf_003.wav
        qaf_004.wav
        qaf_005.wav

    means:

        next recording = 6

    So we no longer need to manually configure:
        START_FROM_LETTER
        START_FROM_NUMBER
    """

    existing_files = sorted(
        letter_dir.glob(f"{folder_name}_*.wav")
    )
    if not existing_files:
        return 1

    numbers = []

    for audio_file in existing_files:
        try:
            number = int(
                audio_file.stem.split("_")[-1]
            )
            numbers.append(number)
        except ValueError:
            continue

    if not numbers:
        return 1

    return max(numbers) + 1


def choose_recording_actions():
    """
    Ask what should happen to the recording.

    A = accept
    P = play
    R = retake
    Q = quit
    """

    while True:
        print()
        print("[A] Accept")
        print("[P] Play")
        print("[R] Retake")
        print("[Q] Quit")

        choice = input("> ").strip().lower()

        if choice in {"a", "p", "r", "q"}:
            return choice

        print("invalid choice.")

def record_letter(
    folder_name: str,
    arabic_letter: str
):
    letter_dir = DATA_DIR / folder_name
    letter_dir.mkdir(exist_ok=True)

    number = get_next_recording_number(letter_dir, folder_name)

    # If this letter already has enough recordings,
    # there is nothing left to collect
    if number > TARGET_RECORDINGS_PER_LETTER:
        print(
            f"{arabic_letter} already has "
            f"{TARGET_RECORDINGS_PER_LETTER} recordings."
        )

        return True

    print()
    print("="*40)
    print(f"Sound: {arabic_letter}")
    print(
        "Starting from recording "
        f"{number}/{TARGET_RECORDINGS_PER_LETTER}"
    )
    print("="*40)

    while number <= TARGET_RECORDINGS_PER_LETTER:

        print()
        print(
            f"{arabic_letter} — recording "
            f"{number}/{TARGET_RECORDINGS_PER_LETTER}"
        )

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
                filename = (
                    letter_dir 
                    / f"{folder_name}_{number:03}.wav"
                )

                write(filename, SAMPLE_RATE, audio)

                print(f"Saved: {filename}")

                number += 1

                break

            elif action == "q":
                print()
                print("Stopping recording session.")

                return False

    print()
    print(
        f"Finished {arabic_letter}: "
        f"{TARGET_RECORDINGS_PER_LETTER} recordings."
    )

    return True

            
def main():
    DATA_DIR.mkdir(exist_ok=True)

    print()
    print("Arabic ML Letter Dataset Recorder")
    print("---------------------------------")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Recording length: {DURATION} seconds")
    print(
        f"Target per sound: "
        f"{TARGET_RECORDINGS_PER_LETTER}"
    )
    print()

    print("Sounds:")
    print()

    for folder_name, arabic_letter in LETTERS.items():
        letter_dir = DATA_DIR / folder_name
        letter_dir.mkdir(exist_ok=True)

        next_number = get_next_recording_number(
            letter_dir,
            folder_name,
        )

        completed = min(
            next_number - 1,
            TARGET_RECORDINGS_PER_LETTER,
        )

        print(
            f"{arabic_letter} "
            f"({folder_name:<12}) "
            f"{completed}/{TARGET_RECORDINGS_PER_LETTER}"
        )

    print()

    for folder_name, arabic_letter in LETTERS.items():

        should_continue = record_letter(
            folder_name,
            arabic_letter,
        )

        if not should_continue:
            print()
            print(
                "Progress has been preserved. "
                "Run the script again to resume."
            )

            return

    print()
    print("Dataset collection complete.")
    print()

    print(
        f"{len(LETTERS)} classes × "
        f"{TARGET_RECORDINGS_PER_LETTER} recordings "
        f"= "
        f"{len(LETTERS) * TARGET_RECORDINGS_PER_LETTER} "
        f"total recordings"
    )

if __name__ == "__main__":
    main()
