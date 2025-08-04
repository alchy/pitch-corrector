import argparse
import os
import torch
import torchaudio
import numpy as np

# Check for librosa availability
try:
    import librosa
    import resampy  # Explicitly check for resampy

    USE_LIBROSA = True
except ImportError as e:
    USE_LIBROSA = False
    print(
        f"Librosa nebo resampy není k dispozici: {e}. Použiji torchaudio pro pitch shift. Doporučuji 'pip install librosa resampy' pro lepší paměťovou efektivitu.")

# Argument parser with detailed help
parser = argparse.ArgumentParser(
    description="""Program pro korekci pitch vstupních WAV souborů na standardní ladění (A4=440 Hz).
    Zpracovává stereo i mono WAV soubory (44.1 kHz, max. 12 sekund) v zadaném vstupním adresáři.
    Detekuje fundamentální frekvenci, koriguje ji na nejbližší MIDI notu a ukládá výstup do výstupního adresáře
    s názvem ve formátu mXXX-NOTA-DbLvl-XX.wav (např. m060-C4-DbLvl-023.wav).

    Příklady použití:
    1. Zpracování adresáře se soubory:
       python pitch_corrector.py --input-dir ./vstup --output-dir ./vystup
       (Zpracuje všechny WAV soubory v adresáři ./vstup a uloží je do ./vystup)
    2. Spuštění nápovědy:
       python pitch_corrector.py --help
    """
)
parser.add_argument(
    '--input-dir',
    required=True,
    help="Cesta k adresáři obsahujícímu vstupní WAV soubory (stereo nebo mono, 44.1 kHz, max. 12 sekund)."
)
parser.add_argument(
    '--output-dir',
    required=True,
    help="Cesta k adresáři, kam se uloží upravené WAV soubory ve formátu mXXX-NOTA-DbLvl-XX.wav."
)
args = parser.parse_args()

# Map MIDI numbers to note names
MIDI_TO_NOTE = {
    0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
}


def midi_to_note_name(midi):
    """Convert MIDI number to note name (e.g., 60 -> C4)"""
    octave = (midi // 12) - 1
    note_idx = midi % 12
    note = MIDI_TO_NOTE[note_idx]
    return f"{note}{octave}"


def freq_to_midi(freq):
    """Convert frequency to MIDI number"""
    return 12 * np.log2(freq / 440.0) + 69


def midi_to_freq(midi):
    """Convert MIDI number to frequency"""
    return 440.0 * 2 ** ((midi - 69) / 12)


def calculate_rms_db(waveform):
    """Calculate RMS loudness in dB, handles mono/stereo"""
    rms = torch.sqrt(torch.mean(waveform ** 2))
    return 20 * torch.log10(rms + 1e-10).item()


def normalize_audio(waveform, target_db=-20):
    """Normalize audio to target dB level for better pitch detection"""
    current_rms = torch.sqrt(torch.mean(waveform ** 2))
    target_rms = 10 ** (target_db / 20)
    scaling_factor = target_rms / (current_rms + 1e-10)
    return waveform * scaling_factor


def preprocess_for_pitch_detection(waveform, sr):
    """Preprocess audio for better pitch detection"""
    # Convert to mono by averaging channels if stereo
    if waveform.shape[0] > 1:
        mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
    else:
        mono_waveform = waveform

    # Normalize to consistent level
    normalized = normalize_audio(mono_waveform, target_db=-20)

    # Apply high-pass filter to remove low-frequency noise
    # Simple high-pass filter using first-order difference
    filtered = torch.zeros_like(normalized)
    filtered[:, 1:] = normalized[:, 1:] - 0.95 * normalized[:, :-1]

    return filtered


def detect_pitch_improved(waveform, sr):
    """Improved pitch detection with preprocessing and multiple methods"""
    # Piano frequency range: A0 (27.5 Hz) to C8 (4186 Hz)
    # Adding some margin for detection accuracy
    PIANO_MIN_FREQ = 25.0  # A0 = 27.5 Hz with margin
    PIANO_MAX_FREQ = 4400.0  # C8 = 4186 Hz with margin

    # Preprocess audio
    processed = preprocess_for_pitch_detection(waveform, sr)

    try:
        # Method 1: Torchaudio's built-in detector
        pitch_frames = torchaudio.functional.detect_pitch_frequency(processed, sr)
        valid_pitches = pitch_frames[pitch_frames > 0]

        if len(valid_pitches) > 0:
            # Use median instead of mean for more robust estimation
            median_pitch = torch.median(valid_pitches).item()

            # Additional validation: check if pitch is in piano range
            if PIANO_MIN_FREQ <= median_pitch <= PIANO_MAX_FREQ:
                return median_pitch

    except Exception as e:
        print(f"Chyba v torchaudio detekci: {e}")

    # Method 2: Autocorrelation-based detection (fallback)
    try:
        if USE_LIBROSA:
            audio_np = processed.numpy().flatten()
            # Use librosa's pitch detection with piano frequency range
            pitches, magnitudes = librosa.piptrack(y=audio_np, sr=sr,
                                                   threshold=0.1,
                                                   fmin=PIANO_MIN_FREQ,
                                                   fmax=PIANO_MAX_FREQ)

            # Find the pitch with highest magnitude in each frame
            pitch_values = []
            for frame in range(pitches.shape[1]):
                idx = magnitudes[:, frame].argmax()
                pitch = pitches[idx, frame]
                if pitch > 0:
                    pitch_values.append(pitch)

            if pitch_values:
                return np.median(pitch_values)

    except Exception as e:
        print(f"Chyba v librosa detekci: {e}")

    # Method 3: Simple autocorrelation fallback
    try:
        audio_flat = processed.flatten()

        # Autocorrelation
        autocorr = torch.nn.functional.conv1d(
            audio_flat.unsqueeze(0).unsqueeze(0),
            audio_flat.flip(0).unsqueeze(0).unsqueeze(0),
            padding=len(audio_flat) - 1
        ).squeeze()

        # Find peak in piano frequency range
        min_period = int(sr / PIANO_MAX_FREQ)  # Max frequency
        max_period = int(sr / PIANO_MIN_FREQ)  # Min frequency

        autocorr_section = autocorr[len(audio_flat):len(audio_flat) + max_period]
        peak_idx = torch.argmax(autocorr_section[min_period:]) + min_period

        fundamental_freq = sr / peak_idx.item()

        if PIANO_MIN_FREQ <= fundamental_freq <= PIANO_MAX_FREQ:
            return fundamental_freq

    except Exception as e:
        print(f"Chyba v autocorrelation detekci: {e}")

    return None


# Process files
input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Používám zařízení: {device}")

for file_name in os.listdir(input_dir):
    if not file_name.endswith('.wav'):
        continue

    # Normalize file path for Windows
    input_path = os.path.abspath(os.path.join(input_dir, file_name)).replace("\\", "/")
    print(f"Zpracovávám {file_name}")

    # Load file
    try:
        waveform, sr = torchaudio.load(input_path)
        original_waveform = waveform.clone()  # Keep original for processing
    except Exception as e:
        print(f"Chyba při načítání {file_name}: {e}")
        continue

    # Validate file
    if sr != 44100:
        print(f"Soubor {file_name} má neplatnou vzorkovací frekvenci {sr} Hz, očekáváno 44100 Hz")
        continue
    if waveform.shape[1] / sr > 12:
        print(f"Soubor {file_name} je delší než 12 sekund")
        continue

    # Detect mono/stereo
    num_channels = waveform.shape[0]
    audio_format = 'stereo' if num_channels == 2 else 'mono'
    print(f"Formát: {audio_format}")

    # Detect fundamental frequency with improved method
    mean_pitch = detect_pitch_improved(waveform, sr)

    if mean_pitch is None:
        print(f"Nelze detekovat platnou frekvenci pro {file_name}, přeskočeno")
        continue

    print(f"Detekovaná frekvence: {mean_pitch:.2f} Hz")

    # Convert to MIDI and round to nearest note
    midi_float = freq_to_midi(mean_pitch)
    midi_int = round(midi_float)
    target_freq = midi_to_freq(midi_int)
    semitone_shift = 12 * np.log2(target_freq / mean_pitch)

    print(f"MIDI: {midi_float:.2f} -> {midi_int}, posun: {semitone_shift:.3f} půltónů")

    # Apply pitch shift
    if abs(semitone_shift) > 0.01:
        try:
            if USE_LIBROSA:
                # Librosa for lower memory usage
                waveform_np = original_waveform.numpy()
                shifted_channels = []
                for ch in range(num_channels):
                    ch_data = waveform_np[ch]
                    shifted = librosa.effects.pitch_shift(
                        y=ch_data, sr=sr, n_steps=semitone_shift, bins_per_octave=12, res_type='kaiser_fast'
                    )
                    shifted_channels.append(shifted)
                waveform = torch.from_numpy(np.array(shifted_channels)).float()
            else:
                # Fallback to torchaudio with optimized parameters
                waveform = original_waveform.to(device)
                transform = torchaudio.transforms.PitchShift(
                    sample_rate=sr, n_steps=semitone_shift, n_fft=512, hop_length=256
                )
                waveform = transform(waveform)
                waveform = waveform.to('cpu')
        except Exception as e:
            print(f"Chyba při pitch shift pro {file_name}: {e}")
            continue
    else:
        waveform = original_waveform
        print("Žádný pitch shift není potřeba")

    # Preserve original loudness
    original_rms = torch.sqrt(torch.mean(original_waveform ** 2))
    current_rms = torch.sqrt(torch.mean(waveform ** 2))
    waveform = waveform / (current_rms + 1e-10) * original_rms

    # Calculate loudness for file name
    db_level = round(calculate_rms_db(waveform))

    # Create output file name
    note_name = midi_to_note_name(midi_int)
    output_file = f"m{midi_int:03d}-{note_name}-DbLvl{db_level:+03d}.wav"
    output_path = os.path.join(output_dir, output_file).replace("\\", "/")

    # Save output
    try:
        torchaudio.save(output_path, waveform, sr)
        print(
            f"Zpracováno: {file_name} -> {output_file}, pitch: {mean_pitch:.2f} Hz -> {target_freq:.2f} Hz, {audio_format}")
        print("-" * 50)
    except Exception as e:
        print(f"Chyba při ukládání {output_file}: {e}")
