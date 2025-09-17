import argparse
import soundfile as sf
import numpy as np
import resampy
from collections import defaultdict
import statistics
from pathlib import Path


class AudioUtils:
    """Utility functions for audio processing"""

    MIDI_TO_NOTE = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    @staticmethod
    def freq_to_midi(freq):
        """Convert frequency to MIDI number"""
        return 12 * np.log2(freq / 440.0) + 69

    @staticmethod
    def midi_to_freq(midi):
        """Convert MIDI number to frequency"""
        return 440.0 * 2 ** ((midi - 69) / 12)

    @staticmethod
    def midi_to_note_name(midi):
        """Convert MIDI number to note name (e.g., 60 -> C4)"""
        octave = (midi // 12) - 1
        note_idx = midi % 12
        note = AudioUtils.MIDI_TO_NOTE[note_idx]
        if len(note) == 1:
            note += '_'  # Pad single characters for consistent formatting
        return f"{note}{octave}"

    @staticmethod
    def calculate_rms_db(waveform):
        """Calculate RMS loudness in dB"""
        rms = np.sqrt(np.mean(waveform ** 2))
        return 20 * np.log10(rms + 1e-10)

    @staticmethod
    def normalize_audio(waveform, target_db=-20.0):
        """Normalize audio to target dB level"""
        rms = np.sqrt(np.mean(waveform ** 2))
        target_rms = 10 ** (target_db / 20)
        return waveform * (target_rms / (rms + 1e-10))


class YINPitchDetector:
    """YIN pitch detection algorithm - optimized for piano samples"""

    def __init__(self, sample_rate=44100, downsample_factor=1):
        self.sample_rate = sample_rate
        self.downsample_factor = downsample_factor
        self.min_duration = 0.1
        self.fmin = 32.0  # Piano range: A0 ≈ 27.5 Hz
        self.fmax = 4186.0  # Piano range: C8 ≈ 4186 Hz
        self.threshold = 0.1
        self.frame_length = 2048
        self.hop_size = 512

    def preprocess(self, waveform):
        """Preprocess waveform for pitch detection"""
        # Convert to mono if stereo
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = np.mean(waveform, axis=1)
        elif len(waveform.shape) > 1:
            waveform = waveform[:, 0]  # Take first channel if already mono

        # Normalize to consistent level
        return AudioUtils.normalize_audio(waveform, target_db=-20.0)

    def yin_difference_function(self, audio, max_tau):
        """YIN difference function"""
        diff = np.zeros(max_tau)
        for tau in range(max_tau):
            if tau == 0:
                diff[tau] = 0
            else:
                diff[tau] = np.sum((audio[:len(audio) - tau] - audio[tau:]) ** 2)
        return diff

    def yin_cmndf(self, diff, max_tau):
        """YIN cumulative mean normalized difference function"""
        cmndf = np.zeros(max_tau)
        cmndf[0] = 1.0
        cumsum = 0.0

        for tau in range(1, max_tau):
            cumsum += diff[tau]
            if cumsum == 0:
                cmndf[tau] = 1.0
            else:
                cmndf[tau] = diff[tau] * tau / cumsum

        return cmndf

    def yin_parabolic_interpolation(self, cmndf, tau):
        """Parabolic interpolation for sub-sample accuracy"""
        if tau <= 0 or tau >= len(cmndf) - 1:
            return tau

        left = cmndf[tau - 1]
        center = cmndf[tau]
        right = cmndf[tau + 1]

        # Parabolic interpolation formula
        denom = 2 * (left - 2 * center + right)
        if abs(denom) < 1e-10:
            return tau

        delta = (left - right) / denom
        return tau + delta / 2

    def yin_pitch_single_frame(self, audio, effective_sr):
        """YIN pitch detection for single frame"""
        max_tau = min(len(audio) - 1, int(effective_sr / self.fmin))
        min_tau = max(1, int(effective_sr / self.fmax))

        if max_tau <= min_tau:
            return None

        # Compute difference function
        diff = self.yin_difference_function(audio, max_tau)

        # Compute CMNDF
        cmndf = self.yin_cmndf(diff, max_tau)

        # Find first minimum below threshold
        for tau in range(min_tau, max_tau):
            if cmndf[tau] < self.threshold:
                # Use parabolic interpolation for better accuracy
                tau_interp = self.yin_parabolic_interpolation(cmndf, tau)
                if tau_interp <= 0:
                    continue
                return effective_sr / tau_interp

        return None

    def detect(self, waveform, verbose=False):
        """Main pitch detection method"""
        duration = len(waveform) / self.sample_rate
        if duration < self.min_duration:
            if verbose:
                print(f"[DEBUG] Signál je příliš krátký ({duration:.3f}s)")
            return None

        # Preprocess audio
        audio = self.preprocess(waveform)

        # Apply downsampling if requested
        effective_sr = self.sample_rate
        if self.downsample_factor > 1:
            if verbose:
                print(f"[DEBUG] Downsampling: faktor {self.downsample_factor}")
            audio = resampy.resample(audio, self.sample_rate, self.sample_rate // self.downsample_factor)
            effective_sr = self.sample_rate // self.downsample_factor

        try:
            pitches = []

            # Analyze multiple frames for stability
            for i in range(0, len(audio) - self.frame_length, self.hop_size):
                frame = audio[i:i + self.frame_length]
                pitch = self.yin_pitch_single_frame(frame, effective_sr)

                if pitch is not None and self.fmin <= pitch <= self.fmax:
                    # Compensate for downsampling
                    compensated_pitch = pitch * self.downsample_factor
                    pitches.append(compensated_pitch)

            if not pitches:
                if verbose:
                    print("[DEBUG] Žádné validní frekvence nebyly detekovány.")
                return None

            # Use median for robustness
            median_pitch = np.median(pitches)

            if verbose:
                print(f"[DEBUG] YIN detekce: {len(pitches)} validních framů")
                print(f"[DEBUG] Detekovaný pitch: {median_pitch:.2f} Hz")

            return median_pitch

        except Exception as e:
            if verbose:
                print(f"[DEBUG] Chyba v YIN detekci: {e}")
            return None


class SimplePitchShifter:
    """Simple pitch shifting via resampling (changes duration)"""

    @staticmethod
    def pitch_shift_simple(audio, sr, semitone_shift):
        """Pitch shift změnou sample rate (mění délku)"""
        if abs(semitone_shift) < 0.01:
            return audio, sr

        factor = 2 ** (semitone_shift / 12)
        new_sr = int(sr * factor)

        try:
            # Handle multi-channel audio
            if len(audio.shape) > 1:
                shifted_channels = []
                for ch in range(audio.shape[1]):
                    shifted = resampy.resample(audio[:, ch], sr, new_sr)
                    shifted_channels.append(shifted)
                shifted_audio = np.column_stack(shifted_channels)
            else:
                shifted_audio = resampy.resample(audio, sr, new_sr)

            return shifted_audio, new_sr

        except Exception as e:
            print(f"[ERROR] Chyba při pitch shift: {e}")
            return audio, sr


class VelocityMapper:
    """Handles velocity mapping and outlier filtering"""

    @staticmethod
    def remove_outliers(rms_values, threshold_db=8.0):
        """Remove outliers based on median absolute deviation"""
        if len(rms_values) < 3:
            return rms_values

        median_rms = statistics.median(rms_values)
        filtered_values = []

        for rms in rms_values:
            if abs(rms - median_rms) <= threshold_db:
                filtered_values.append(rms)

        return filtered_values if filtered_values else rms_values

    @staticmethod
    def create_velocity_mapping(rms_values):
        """Create linear velocity mapping (0-7) from RMS values"""
        if len(rms_values) <= 1:
            return {0: rms_values}

        min_rms = min(rms_values)
        max_rms = max(rms_values)

        # Handle case where all values are very similar
        if abs(max_rms - min_rms) < 0.1:
            return {0: rms_values}

        # Linear mapping to velocity 0-7
        velocity_map = defaultdict(list)

        for rms in rms_values:
            if max_rms == min_rms:
                velocity = 0
            else:
                velocity = round(7 * (rms - min_rms) / (max_rms - min_rms))
                velocity = max(0, min(7, velocity))

            velocity_map[velocity].append(rms)

        return dict(velocity_map)


class AudioSample:
    """Container for audio sample data"""

    def __init__(self, filepath, waveform, sr, rms_db, midi_note):
        self.filepath = filepath
        self.waveform = waveform
        self.sr = sr
        self.rms_db = rms_db
        self.midi_note = midi_note
        self.velocity = None


class PitchCorrectorWithVelocityMapping:
    """Main processor class"""

    def __init__(self, input_dir, output_dir, outlier_threshold=8.0, verbose=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.outlier_threshold = outlier_threshold
        self.verbose = verbose
        self.max_semitone_shift = 1.0  # Maximum ±1 semitone

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.pitch_detector = YINPitchDetector()
        self.pitch_shifter = SimplePitchShifter()
        self.velocity_mapper = VelocityMapper()

    def generate_unique_filename(self, midi, velocity, sample_rate_str):
        """Generate unique filename with -next1, -next2, etc."""
        base_filename = f"m{midi:03d}-vel{velocity}-{sample_rate_str}"
        output_path = self.output_dir / f"{base_filename}.wav"

        if not output_path.exists():
            return output_path

        counter = 1
        while True:
            filename = f"{base_filename}-next{counter}.wav"
            output_path = self.output_dir / filename
            if not output_path.exists():
                return output_path
            counter += 1

    def load_and_analyze_files(self):
        """Phase 1: Load and analyze all files"""
        print("\n=== FÁZE 1: Načítání a analýza souborů ===")
        samples = []

        wav_files = list(self.input_dir.glob("*.wav")) + list(self.input_dir.glob("*.WAV"))

        if not wav_files:
            print("Nebyly nalezeny žádné WAV soubory!")
            return samples

        for file_path in wav_files:
            print(f"\nZpracovávám: {file_path.name}")
            print(f"[DEBUG] Čtu soubor: {file_path}")

            try:
                waveform, sr = sf.read(str(file_path))
                # Ensure waveform is 2D (samples, channels)
                if len(waveform.shape) == 1:
                    waveform = waveform[:, np.newaxis]
                print(f"[DEBUG] Úspěšně načten: {waveform.shape[0]} vzorků, {waveform.shape[1]} kanálů, {sr} Hz")
            except Exception as e:
                print(f"[ERROR] Chyba při načítání {file_path.name}: {e}")
                continue

            # Validate sample rate
            if sr not in [44100, 48000]:
                print(f"[WARNING] Nepodporovaná vzorkovací frekvence {sr} Hz pro {file_path.name}")
                continue

            # Calculate duration (no limit check)
            duration = len(waveform) / sr

            # Update pitch detector sample rate
            self.pitch_detector.sample_rate = sr

            # Detect pitch using YIN (convert to 1D for detection)
            waveform_1d = waveform[:, 0] if waveform.shape[1] > 1 else waveform.flatten()
            detected_pitch = self.pitch_detector.detect(waveform_1d, verbose=self.verbose)

            if detected_pitch is None:
                print(f"[WARNING] Nelze detekovat pitch pro {file_path.name}")
                continue

            midi_float = AudioUtils.freq_to_midi(detected_pitch)

            # Find nearest MIDI note (not just rounding)
            midi_lower = int(np.floor(midi_float))
            midi_upper = int(np.ceil(midi_float))

            # Calculate distances to both notes
            dist_lower = abs(midi_float - midi_lower)
            dist_upper = abs(midi_float - midi_upper)

            # Choose the one with smaller correction needed
            if dist_lower <= dist_upper:
                midi_int = midi_lower
                semitone_diff = midi_float - midi_lower
            else:
                midi_int = midi_upper
                semitone_diff = midi_float - midi_upper
            if abs(semitone_diff) > self.max_semitone_shift:
                print(
                    f"[WARNING] Soubor {file_path.name} vyžaduje korekci {semitone_diff:.2f} půltónů (>±{self.max_semitone_shift}), zahazuji")
                continue

            # Calculate RMS
            rms_db = AudioUtils.calculate_rms_db(waveform)

            sample = AudioSample(file_path, waveform, sr, rms_db, midi_int)
            samples.append(sample)

            note_name = AudioUtils.midi_to_note_name(midi_int)
            print(f"  Pitch: {detected_pitch:.2f} Hz → MIDI {midi_int} ({note_name})")
            print(f"  Korekce: {semitone_diff:.3f} půltónů")
            print(f"  RMS: {rms_db:.2f} dB, SR: {sr} Hz, délka: {duration:.2f}s")

        print(f"\nCelkem načteno {len(samples)} zpracovatelných vzorků")
        return samples

    def create_velocity_mappings(self, samples):
        """Phase 2: Create velocity mappings per MIDI note"""
        print("\n=== FÁZE 2: Tvorba velocity map ===")

        # Group samples by MIDI note
        midi_groups = defaultdict(list)
        for sample in samples:
            midi_groups[sample.midi_note].append(sample)

        velocity_mappings = {}

        for midi_note, midi_samples in midi_groups.items():
            note_name = AudioUtils.midi_to_note_name(midi_note)
            rms_values = [s.rms_db for s in midi_samples]

            print(f"\nMIDI {midi_note} ({note_name}): {len(midi_samples)} vzorků")
            print(f"  RMS rozsah: {min(rms_values):.2f} až {max(rms_values):.2f} dB")

            # Remove outliers
            filtered_rms = self.velocity_mapper.remove_outliers(rms_values, self.outlier_threshold)
            outliers_removed = len(rms_values) - len(filtered_rms)

            if outliers_removed > 0:
                print(f"  Odstraněno {outliers_removed} outlierů")

            # Create velocity mapping
            velocity_map = self.velocity_mapper.create_velocity_mapping(filtered_rms)
            velocity_mappings[midi_note] = velocity_map

            # Assign velocity to samples
            for sample in midi_samples:
                sample.velocity = None
                for vel, rms_list in velocity_map.items():
                    if sample.rms_db in rms_list:
                        sample.velocity = vel
                        break

            print(f"  Velocity mapping:")
            for vel, rms_list in sorted(velocity_map.items()):
                print(f"    vel{vel}: {len(rms_list)} vzorků (RMS {min(rms_list):.1f} až {max(rms_list):.1f} dB)")

        return velocity_mappings

    def process_and_export(self, samples):
        """Phase 3: Process and export files"""
        print("\n=== FÁZE 3: Tuning a export ===")

        total_outputs = 0

        for sample in samples:
            if sample.velocity is None:
                continue  # Skip outliers

            midi_note = sample.midi_note
            note_name = AudioUtils.midi_to_note_name(midi_note)

            print(f"\nZpracovávám: {sample.filepath.name}")
            print(f"  MIDI {midi_note} ({note_name}) → velocity {sample.velocity}")

            # Pitch correction
            target_freq = AudioUtils.midi_to_freq(midi_note)

            # Re-detect pitch for accurate correction
            waveform_1d = sample.waveform[:, 0] if sample.waveform.shape[1] > 1 else sample.waveform.flatten()
            detected_pitch = self.pitch_detector.detect(waveform_1d, verbose=False)

            if detected_pitch:
                semitone_shift = 12 * np.log2(target_freq / detected_pitch)

                # Double-check shift limit
                if abs(semitone_shift) > self.max_semitone_shift:
                    print(f"  [WARNING] Korekce {semitone_shift:.3f} půltónů překračuje limit, přeskakuji")
                    continue

                print(f"  Pitch korekce: {detected_pitch:.2f} Hz → {target_freq:.2f} Hz ({semitone_shift:.3f} půltónů)")

                # Apply simple pitch shift (changes duration)
                tuned_waveform, tuned_sr = self.pitch_shifter.pitch_shift_simple(
                    sample.waveform, sample.sr, semitone_shift
                )

                # Calculate new duration
                original_duration = len(sample.waveform) / sample.sr
                new_duration = len(tuned_waveform) / tuned_sr
                print(f"  Délka: {original_duration:.3f}s → {new_duration:.3f}s")

            else:
                tuned_waveform = sample.waveform
                tuned_sr = sample.sr
                print(f"  Bez pitch korekce (re-detection failed)")

            # Generate outputs for both target sample rates
            target_sample_rates = [(44100, 'f44'), (48000, 'f48')]

            for target_sr, sr_suffix in target_sample_rates:
                # Convert sample rate if needed
                if tuned_sr != target_sr:
                    try:
                        if len(tuned_waveform.shape) > 1 and tuned_waveform.shape[1] > 1:
                            # Multi-channel
                            converted_channels = []
                            for ch in range(tuned_waveform.shape[1]):
                                converted = resampy.resample(tuned_waveform[:, ch], tuned_sr, target_sr)
                                converted_channels.append(converted)
                            output_waveform = np.column_stack(converted_channels)
                        else:
                            # Mono
                            output_waveform = resampy.resample(tuned_waveform.flatten(), tuned_sr, target_sr)
                            output_waveform = output_waveform[:, np.newaxis]
                    except Exception as e:
                        print(f"  [ERROR] Chyba při konverzi sample rate: {e}")
                        continue
                else:
                    output_waveform = tuned_waveform

                # Generate unique filename
                output_path = self.generate_unique_filename(midi_note, sample.velocity, sr_suffix)
                print(f"[DEBUG] Připravuji zápis: {output_path}")

                # Save file
                try:
                    sf.write(str(output_path), output_waveform, target_sr)
                    print(f"  Uložen: {output_path.name}")
                    print(f"[DEBUG] Úspěšně zapsán: {output_path} ({len(output_waveform)} vzorků, {target_sr} Hz)")
                    total_outputs += 1
                except Exception as e:
                    print(f"  [ERROR] Chyba při ukládání {output_path.name}: {e}")
                    print(f"[DEBUG] Neúspěšný zápis do: {output_path}")

        return total_outputs

    def process_all(self):
        """Main processing pipeline"""
        print(f"Vstupní adresář: {self.input_dir}")
        print(f"Výstupní adresář: {self.output_dir}")
        print(f"Outlier threshold: {self.outlier_threshold} dB")
        print(f"Max pitch korekce: ±{self.max_semitone_shift} půltónů")

        # Phase 1: Load and analyze
        samples = self.load_and_analyze_files()
        if not samples:
            return

        # Phase 2: Create velocity mappings
        velocity_mappings = self.create_velocity_mappings(samples)

        # Phase 3: Process and export
        total_outputs = self.process_and_export(samples)

        print(f"\n=== DOKONČENO ===")
        print(f"Celkem vytvořeno {total_outputs} výstupních souborů")
        print(f"Výstupní adresář: {self.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Program pro korekci pitch a velocity mapping s YIN algoritmem a jednoduchým pitch shiftem.
        Zpracovává WAV soubory (44.1 kHz a 48 kHz) s maximální korekcí ±1 půltón.
        Vytváří velocity mapy (0-7) podle RMS hlasitosti pro každou MIDI notu.
        Ukládá výstup v obou formátech s názvem ve formátu mXXX-velY-fZZ.wav.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--input-dir', required=True,
                        help='Cesta k adresáři s vstupními WAV soubory')
    parser.add_argument('--output-dir', required=True,
                        help='Cesta k výstupnímu adresáři')
    parser.add_argument('--outlier-threshold', type=float, default=8.0,
                        help='Práh pro odstranění outlierů v dB (výchozí: 8.0)')
    parser.add_argument('--verbose', action='store_true',
                        help='Podrobný výstup pro debugging')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    corrector = PitchCorrectorWithVelocityMapping(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        outlier_threshold=args.outlier_threshold,
        verbose=args.verbose
    )

    corrector.process_all()