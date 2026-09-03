import torch
import torch.nn.functional as F
import torchaudio
from scipy.io.wavfile import read

SAMPLE_RATE = 16_000

TARGET_DURATION = 1.2

TARGET_NUM_SAMPLES = int(
    SAMPLE_RATE * TARGET_DURATION
)


# When deciding whether a part of the waveform contains
# actual speech, treat samples above 2% of this recording's
# peak amplitude as potentially active.

# This is a RELATIVE threshold:

# quiet recording:
#     lower threshold

# loud recording:
#     higher threshold

# which is more useful than one fixed amplitude threshold
# for every recording.
SILENCE_THRESHOLD_RATIO = 0.02


# Keep 50 ms of audio before and after the detected sound.

# We do not want to trim exactly to the first loud sample
# because quieter consonant information around the beginning
# or end may still be important.
TRIM_MARGIN_SECONDS = 0.05

TRIM_MARGIN_SAMPLES = int(
    SAMPLE_RATE * TRIM_MARGIN_SECONDS
)


# FTT: Fast Fourier Transform
    #   (It tells us which frequencies are contained in this piece of waveform?)
    # FTT might discover 200 Hz -> some energy but 500Hz -> lots of energy
    # But speech changes constantly -> We therefore don't analyse the whole 1.2 seconds at once.
    # We chop it into lots of overlapping windows:

    # at 16kHz 1024 samples ÷ 16000 samples/sec = 0.064 seconds
    # Our current configuration therefore analyses chunks of roughly 64 ms.

    # important trade-off:
        # larger window
            # → better frequency detail
            # → poorer time detail

        # smaller window
            # → better time detail
            # → poorer frequency detail

# hop_length: How far should I move forward before analysing the next chunk?
    # At 16kHz: 256 / 16000 = 0.016 seconds
    # so 16ms -> the chucks overlap heavily ->
    # without overlapping we could miss important transitions between sounds
    # this is why the spectrogram ends up with around 75 time positions over the 1.2s recording.

# n_mels:
    # Convert all those raw FFT frequency bins into 64 Mel-frequency bands.
    # FTT of 1024 gives us 513 useful frequency bins
    # so 513 frequency bins to 64 mel bands
    # so the Mel scale compresses the frequency axis into
    # something more aligned with human auditory perception.

# 1   = audio channel
# 64  = Mel frequency bands
# 75  = moments in time

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=512,
    win_length=400,
    hop_length=160,
    n_mels=64,
)

# n_fft=512
#     FFT calculation size

# win_length=400
#     actually examine 400 samples
#     = 25 ms of speech

# hop_length=160
#     move forward 160 samples
#     = 10 ms

# n_mels=64
#     represent the frequency spectrum
#     using 64 Mel bands

db_transform = torchaudio.transforms.AmplitudeToDB()


def load_wav(path):
    sample_rate, audio = read(path)

    # audio is a numpy array containing the microphone measurements
    # 1.2 seconds X 16,000 sample rate = 19,200 samples

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"{path} has sample rate {sample_rate}, "
            f"expected {SAMPLE_RATE}"
        )

    # NumPy array -> torch.from_numpy() -> PyTorch Tensor
    # .float() ensures the numbers are stored as 32-bit floating-point values
    waveform = torch.from_numpy(audio).float()

    # Pytorch audio tool expects audio in the format of
    #   [channels, samples]
    # for our mono audio that would be: [1, 19200]
    # But WAV-reading libraries don't always return the dimensions in that format.
    
    if waveform.ndim == 1:
        # Case A: NumPy gives us this
        #   (19200,)
        # so waveform.unsqueeze(0) changes (19200,) -> (1, 19200)
        waveform = waveform.unsqueeze(0)
    else:
        # Case B: NumPy gives us this
        #   (19200, 1) meaning [samples, channels]
        # TorchAudio wants: [channels, samples]
        # so waveform.transpose(0, 1) turns (19200, 1) into (1, 19200)
        waveform = waveform.transpose(0, 1)

    return waveform

def trim_and_center_waveform(waveform):
    # SILENCE TRIMMING + CENTRING
        
        # Our recordings are all 1.2 seconds long, but we did not
        # pronounce each sound at exactly the same moment.
        
        # For example:
            
            # Recording A:
            # [................قَ......]
            
            # Recording B:
            # [......قَ................]
        
        # The actual Arabic sound might be equivalent, but the CNN
        # can potentially learn the POSITION of the sound as an
        # accidental shortcut.
        
        # We therefore:
            
            # 1. detect approximately where the actual sound begins
            #    and ends;
            
            # 2. remove excessive surrounding silence;
            
            # 3. place the detected sound in the centre of a new
            #    1.2-second waveform.
        
        # We do NOT:
            # - change pitch
            # - change speed
            # - stretch the pronunciation
        
        # We are only changing where the existing sound sits
        # inside the recording window.


    # For every time sample, find its absolute amplitude.
    
    # waveform shape:
        # [channels, samples]
    
    # amplitude shape:
        # [samples]
    amplitude = waveform.abs().amax(
        dim=0
    )


    # Find the loudest sample in this recording.
    peak_amplitude = amplitude.max()


    # Protect against a completely silent recording.
    if peak_amplitude <= 1e-8:
        return torch.zeros(
            (
                waveform.shape[0],
                TARGET_NUM_SAMPLES,
            ),
            dtype=waveform.dtype,
        )


    # Instead of using one fixed absolute threshold,
    # make the threshold relative to this recording.
    
    # 0.02 means:
        # 2% of this recording's peak amplitude.
    
    # Anything above this is treated as potentially
    # belonging to the actual pronunciation.
    threshold = (
        peak_amplitude
        * SILENCE_THRESHOLD_RATIO
    )


    active_samples = torch.nonzero(
        amplitude >= threshold,
        as_tuple=False,
    ).flatten()


    # This should rarely happen because we already checked
    # for a silent recording, but keep the function safe.
    if active_samples.numel() == 0:
        return waveform


    first_active = active_samples[0].item()
    last_active = active_samples[-1].item()


    # Preserve a small amount of audio before and after
    # the detected active region.
    
    # This matters because the beginning/release of a consonant
    # may contain quiet but important acoustic information.
    start = max(
        0,
        first_active - TRIM_MARGIN_SAMPLES,
    )

    end = min(
        waveform.shape[1],
        last_active + TRIM_MARGIN_SAMPLES + 1,
    )


    trimmed = waveform[
        :,
        start:end,
    ]


    # The trimmed pronunciation is normally shorter than
    # our fixed 1.2-second input.
    
    # Work out how much silence needs to be added back.
    padding_needed = (
        TARGET_NUM_SAMPLES
        - trimmed.shape[1]
    )


    # Safety case:
    # if an active recording somehow exceeds our target size,
    # take the central TARGET_NUM_SAMPLES samples.
    if padding_needed < 0:

        excess = (
            trimmed.shape[1]
            - TARGET_NUM_SAMPLES
        )

        crop_start = excess // 2

        return trimmed[
            :,
            crop_start:
            crop_start + TARGET_NUM_SAMPLES,
        ]


    # Split the required silence approximately equally
    # between the left and right sides.
    left_padding = padding_needed // 2

    right_padding = (
        padding_needed
        - left_padding
    )


    centred = F.pad(
        trimmed,
        (
            left_padding,
            right_padding,
        ),
    )

    return centred

def augment_waveform(waveform):
    # DATA AUGMENTATION:
        # The purpose of augmentation is to make it harder for the
        # neural network to simply memorise the exact 60 training recordings.
        
        # Each time a training recording is loaded, we can make small,
        # random changes which should NOT change which Arabic letter it is.
        
        # For example:
            # original قَ
            #     ↓
            # slightly quieter/louder
            # slightly earlier/later
            # small amount of background noise
            #     ↓
            # still قَ
        
        # This encourages the CNN to learn features which remain consistent
        # across different recordings of the same sound instead of learning
        # unrelated patterns specific to one WAV file.



    # 1. RANDOM VOLUME / GAIN

    # Choose a random multiplier between 0.8 and 1.2.
    
    # Examples:
        # 0.8 → 80% of the original amplitude
        # 1.0 → unchanged
        # 1.2 → 120% of the original amplitude
    
    # The pronunciation is still the same letter.
    gain = torch.empty(1).uniform_(0.8, 1.2).item()

    waveform = waveform * gain

    # 2. RANDOM TIME SHIFT

    # Maximum shift:
        # 800 audio samples

        # At 16,000 samples per second:
            # 800 / 16,000
            # = 0.05 seconds
            # = 50 ms
    
    # So the pronunciation can randomly move up to:
        # 50 ms earlier
        # or
        # 50 ms later

    # This stops the model relying too strongly on exactly where
    # inside the 1.2-second recording the pronunciation begins.

    max_shift = 800

    shift = torch.randint(
        -max_shift,
        max_shift + 1,
        (1,),
    ).item()

    if shift > 0:
        # Move the waveform later.
        
        # Example:
            # [sound........silence]
            #          ↓
            # [silence.sound.......]
        
        # Add zeros to the beginning and remove the same number
        # of samples from the end so the total length stays unchanged.
        waveform = torch.nn.functional.pad(
            waveform,
            (shift, 0),
        )[:, :-shift]

    elif shift < 0:
        # Move the waveform earlier.

        # Remove samples from the beginning and add zeros to the end.
        shift = abs(shift)

        waveform = torch.nn.functional.pad(
            waveform[:, shift:],
            (0, shift),
        )


    # 3. RANDOM BACKGROUND NOISE

    # Measure approximately how much the existing waveform varies.
    signal_std = waveform.std()

    # Choose random noise strength between:
        # 0% and 5% of the signal's standard deviation.
    
    # Sometimes this will add almost no noise.
    # Other times it will add a small amount.
    noise_strength = torch.empty(1).uniform_(0.0, 0.05).item()

    noise = (
        torch.randn_like(waveform)
        * signal_std
        * noise_strength
    )

    waveform = waveform + noise


    # Keep audio values inside the normal floating-point audio range.
    waveform = waveform.clamp(-1.0, 1.0)

    return waveform


def waveform_to_mel(waveform):
    # Convert raw waveform into a Mel spectrogram.

    # mel = self.mel_transform(waveform) is essentially:
    # raw waveform
    #     ↓
    # split into overlapping windows
    #     ↓
    # FFT each window
    #     ↓
    # calculate frequency power
    #     ↓
    # convert frequencies to Mel bands
    #     ↓
    # Mel spectrogram
    mel = mel_transform(waveform)

    # Convert the power values to a logarithmic decibel scale.
    # log-scaled spectrograms are often a very useful representation for speech/audio ML.
    mel_db = db_transform(mel)

    # Normalise this spectrogram so that it has approximately:
    #
    # mean = 0
    # standard deviation = 1
    #
    # This gives the CNN a more consistent numerical input range.
    mean = mel_db.mean()
    std = mel_db.std()

    mel_db = (mel_db - mean) / (std + 1e-6)

    return mel_db