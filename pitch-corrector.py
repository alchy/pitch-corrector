"""
Program pro korekci pitch a velocity mapping vzorků elektrického piana
s pokročilou YIN detekcí a adaptivní oktávovou korekcí.

Autor: Refaktorovaná verze pro Vintage Vibe elektrické piano
Datum: 2025
"""

import argparse
import soundfile as sf
import numpy as np
import resampy
from collections import defaultdict
import statistics
from pathlib import Path


class AudioUtils:
    """Pomocné funkce pro práci s audio daty"""

    MIDI_TO_NOTE = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    @staticmethod
    def freq_to_midi(freq):
        """Převod frekvence na MIDI číslo"""
        return 12 * np.log2(freq / 440.0) + 69

    @staticmethod
    def midi_to_freq(midi):
        """Převod MIDI čísla na frekvenci"""
        return 440.0 * 2 ** ((midi - 69) / 12)

    @staticmethod
    def midi_to_note_name(midi):
        """Převod MIDI čísla na název noty (např. 60 -> C4)"""
        octave = (midi // 12) - 1
        note_idx = midi % 12
        note = AudioUtils.MIDI_TO_NOTE[note_idx]
        if len(note) == 1:
            note += '_'  # Padding pro konzistentní formátování
        return f"{note}{octave}"

    @staticmethod
    def calculate_rms_db(waveform):
        """Výpočet RMS hlasitosti v dB"""
        rms = np.sqrt(np.mean(waveform ** 2))
        return 20 * np.log10(rms + 1e-10)

    @staticmethod
    def normalize_audio(waveform, target_db=-20.0):
        """Normalizace audio na cílovou úroveň v dB"""
        rms = np.sqrt(np.mean(waveform ** 2))
        target_rms = 10 ** (target_db / 20)
        return waveform * (target_rms / (rms + 1e-10))


class AdvancedYINDetector:
    """
    Pokročilý YIN pitch detector s adaptivními parametry a oktávovou korekcí.
    Optimalizován pro elektrické piano vzorky s dynamickým envelope.
    """

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.min_duration = 0.1

        # Piano rozsah frekvencí
        self.fmin = 27.5  # A0
        self.fmax = 4186.0  # C8

        # YIN parametry - adaptivní podle sample rate
        self.threshold = 0.1
        self.update_parameters_for_sample_rate(sample_rate)

        # Oktávové hranice pro korekci
        self.low_octave_threshold = 146.83  # D3 - pod touto frekvencí použij upsampling
        self.high_octave_threshold = 659.25  # E5 - nad touto frekvencí použij downsampling

    def update_parameters_for_sample_rate(self, sample_rate):
        """Aktualizuje YIN parametry podle vzorkovací frekvence"""
        self.sample_rate = sample_rate

        # Adaptivní velikosti oken podle sample rate
        if sample_rate >= 48000:
            self.frame_length = 2048  # ~43ms při 48kHz
            self.hop_size = 512  # ~11ms při 48kHz
        else:  # 44100 Hz
            self.frame_length = 2048  # ~46ms při 44.1kHz
            self.hop_size = 512  # ~12ms při 44.1kHz

        print(f"[DEBUG] YIN parametry pro {sample_rate}Hz: frame={self.frame_length}, hop={self.hop_size}")

    def find_sustain_section(self, waveform, sustain_ratio=0.6):
        """
        Nalezne sustain sekci vzorku s nejvyšší RMS energií.
        Vynechá attack a release části pro stabilnější pitch detekci.
        """
        if len(waveform) < self.frame_length * 4:
            # Příliš krátký vzorek - použij celý
            return waveform

        # Rozdělení na segmenty pro RMS analýzu
        segment_length = len(waveform) // 10  # 10 segmentů
        rms_values = []

        for i in range(10):
            start = i * segment_length
            end = min((i + 1) * segment_length, len(waveform))
            segment = waveform[start:end]
            rms = np.sqrt(np.mean(segment ** 2))
            rms_values.append((rms, start, end))

        # Najdi segmenty s nejvyšší RMS (sustain oblast)
        rms_values.sort(key=lambda x: x[0], reverse=True)

        # Vezmi top segmenty představující sustain_ratio celkové délky
        total_sustain_samples = int(len(waveform) * sustain_ratio)
        selected_segments = []
        current_samples = 0

        for rms, start, end in rms_values:
            if current_samples >= total_sustain_samples:
                break
            selected_segments.append((start, end))
            current_samples += (end - start)

        # Seřaď segmenty podle pozice a spojí je
        selected_segments.sort()
        sustain_audio = []

        for start, end in selected_segments:
            sustain_audio.extend(waveform[start:end])

        sustain_audio = np.array(sustain_audio)

        print(
            f"[DEBUG] Sustain detekce: {len(sustain_audio)}/{len(waveform)} vzorků ({len(sustain_audio) / len(waveform) * 100:.1f}%)")
        return sustain_audio

    def preprocess_for_octave(self, waveform, rough_pitch):
        """
        Pre-processing pro oktávovou korekci podle hrubé pitch detekce.
        Posune extrémní frekvence do optimálního detekčního pásma.
        """
        original_length = len(waveform)

        if rough_pitch > self.high_octave_threshold:
            # Vysoké tóny - downsample (posun dolů)
            octave_shift = 0
            test_pitch = rough_pitch

            while test_pitch > self.high_octave_threshold and octave_shift < 3:
                test_pitch /= 2
                octave_shift += 1

            if octave_shift > 0:
                # Downsample pro nižší efektivní frekvenci
                new_sr = self.sample_rate // (2 ** octave_shift)
                processed_audio = resampy.resample(waveform, self.sample_rate, new_sr)

                print(
                    f"[DEBUG] Vysoký tón korekce: {rough_pitch:.1f}Hz -> shift -{octave_shift} oktáv, nový SR: {new_sr}Hz")
                return processed_audio, new_sr, octave_shift, True

        elif rough_pitch < self.low_octave_threshold:
            # Nízké tóny - upsample (posun nahoru) 
            octave_shift = 0
            test_pitch = rough_pitch

            while test_pitch < self.low_octave_threshold and octave_shift < 2:
                test_pitch *= 2
                octave_shift += 1

            if octave_shift > 0:
                # Upsample pro vyšší efektivní frekvenci
                new_sr = self.sample_rate * (2 ** octave_shift)
                processed_audio = resampy.resample(waveform, self.sample_rate, new_sr)

                print(
                    f"[DEBUG] Nízký tón korekce: {rough_pitch:.1f}Hz -> shift +{octave_shift} oktáv, nový SR: {new_sr}Hz")
                return processed_audio, new_sr, octave_shift, False

        # Žádná korekce potřebná
        return waveform, self.sample_rate, 0, None

    def preprocess_audio(self, waveform):
        """Základní pre-processing audio dat"""
        # Převod na mono pokud je stereo
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = np.mean(waveform, axis=1)
        elif len(waveform.shape) > 1:
            waveform = waveform[:, 0]

        # Normalizace na konzistentní úroveň
        return AudioUtils.normalize_audio(waveform, target_db=-20.0)

    def yin_difference_function(self, audio, max_tau):
        """YIN difference function - základní součást algoritmu"""
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
        """Parabolická interpolace pro sub-sample přesnost"""
        if tau <= 0 or tau >= len(cmndf) - 1:
            return tau

        left = cmndf[tau - 1]
        center = cmndf[tau]
        right = cmndf[tau + 1]

        # Parabolická interpolační formula
        denom = 2 * (left - 2 * center + right)
        if abs(denom) < 1e-10:
            return tau

        delta = (left - right) / denom
        return tau + delta / 2

    def yin_pitch_single_frame(self, audio, effective_sr):
        """YIN pitch detekce pro jeden frame"""
        max_tau = min(len(audio) - 1, int(effective_sr / self.fmin))
        min_tau = max(1, int(effective_sr / self.fmax))

        if max_tau <= min_tau:
            return None

        # Výpočet difference function
        diff = self.yin_difference_function(audio, max_tau)

        # Výpočet CMNDF
        cmndf = self.yin_cmndf(diff, max_tau)

        # Najdi první minimum pod prahem
        for tau in range(min_tau, max_tau):
            if cmndf[tau] < self.threshold:
                # Použij parabolickou interpolaci pro lepší přesnost
                tau_interp = self.yin_parabolic_interpolation(cmndf, tau)
                if tau_interp <= 0:
                    continue
                return effective_sr / tau_interp

        return None

    def basic_yin_detection(self, audio, effective_sr):
        """Základní YIN detekce bez adaptivních vylepšení"""
        try:
            pitches = []

            # Analýza více framů pro stabilitu
            for i in range(0, len(audio) - self.frame_length, self.hop_size):
                frame = audio[i:i + self.frame_length]
                pitch = self.yin_pitch_single_frame(frame, effective_sr)

                if pitch is not None and self.fmin <= pitch <= self.fmax:
                    pitches.append(pitch)

            if not pitches:
                return None

            # Použij medián pro robustnost
            return np.median(pitches)

        except Exception as e:
            print(f"[DEBUG] Chyba v basic YIN detekci: {e}")
            return None

    def detect(self, waveform, verbose=False):
        """
        Hlavní metoda pro pokročilou pitch detekci s adaptivními vylepšeními.

        Workflow:
        1. Pre-processing (mono, normalizace)
        2. Nalezení sustain sekce
        3. Hrubá detekce pro oktávovou korekci
        4. Adaptivní oktávové předzpracování
        5. Finální přesná detekce
        """
        duration = len(waveform) / self.sample_rate
        if duration < self.min_duration:
            if verbose:
                print(f"[DEBUG] Signál je příliš krátký ({duration:.3f}s)")
            return None

        if verbose:
            print(f"[DEBUG] Začínám pokročilou YIN detekci pro {duration:.2f}s audio")

        # 1. Základní pre-processing
        audio = self.preprocess_audio(waveform)

        # 2. Nalezení sustain sekce pro stabilnější detekci
        sustain_audio = self.find_sustain_section(audio)

        # 3. Hrubá detekce pro určení oktávové korekce
        rough_pitch = self.basic_yin_detection(sustain_audio, self.sample_rate)

        if rough_pitch is None:
            if verbose:
                print("[DEBUG] Hrubá detekce selhala")
            return None

        if verbose:
            print(f"[DEBUG] Hrubá detekce: {rough_pitch:.2f} Hz")

        # 4. Adaptivní oktávové předzpracování
        processed_audio, effective_sr, octave_shift, shift_direction = self.preprocess_for_octave(
            sustain_audio, rough_pitch
        )

        # 5. Finální přesná detekce
        final_pitch = self.basic_yin_detection(processed_audio, effective_sr)

        if final_pitch is None:
            if verbose:
                print("[DEBUG] Finální detekce selhala, používám hrubou detekci")
            return rough_pitch

        # 6. Korekce výsledku podle oktávového posunu
        if octave_shift > 0:
            if shift_direction is True:  # Byl downsample (vysoké tóny)
                corrected_pitch = final_pitch * (2 ** octave_shift)
            elif shift_direction is False:  # Byl upsample (nízké tóny)  
                corrected_pitch = final_pitch / (2 ** octave_shift)
            else:
                corrected_pitch = final_pitch
        else:
            corrected_pitch = final_pitch

        if verbose:
            print(f"[DEBUG] Finální pitch: {corrected_pitch:.2f} Hz (korekce: {octave_shift} oktáv)")

        # Validace finálního výsledku
        if self.fmin <= corrected_pitch <= self.fmax:
            return corrected_pitch
        else:
            if verbose:
                print(f"[DEBUG] Korigovaný pitch {corrected_pitch:.2f} Hz je mimo rozsah, používám hrubou detekci")
            return rough_pitch


class SimplePitchShifter:
    """Jednoduchý pitch shifting pomocí resample (mění délku)"""

    @staticmethod
    def pitch_shift_simple(audio, sr, semitone_shift):
        """
        Pitch shift změnou sample rate - mění délku vzorku.
        Vhodné pro piano samples kde kratší vysoké/delší nízké tóny jsou realistické.
        """
        if abs(semitone_shift) < 0.01:
            return audio, sr

        factor = 2 ** (semitone_shift / 12)
        new_sr = int(sr * factor)

        try:
            # Zpracování multi-channel audio
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
    """Správa velocity mappingu a filtrování outlierů"""

    @staticmethod
    def remove_outliers(rms_values, threshold_db=8.0):
        """Odstranění outlierů na základě mediánové absolutní odchylky"""
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
        """Vytvoření lineárního velocity mappingu (0-7) z RMS hodnot"""
        if len(rms_values) <= 1:
            return {0: rms_values}

        min_rms = min(rms_values)
        max_rms = max(rms_values)

        # Ošetření případu velmi podobných hodnot
        if abs(max_rms - min_rms) < 0.1:
            return {0: rms_values}

        # Lineární mapování do velocity 0-7
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
    """Container pro data audio vzorku"""

    def __init__(self, filepath, waveform, sr, rms_db, midi_note):
        self.filepath = filepath
        self.waveform = waveform
        self.sr = sr
        self.rms_db = rms_db
        self.midi_note = midi_note
        self.velocity = None


class PitchCorrectorWithVelocityMapping:
    """
    Hlavní třída procesoru pro korekci pitch a velocity mapping.
    Optimalizována pro elektrické piano vzorky.
    """

    def __init__(self, input_dir, output_dir, outlier_threshold=8.0, verbose=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.outlier_threshold = outlier_threshold
        self.verbose = verbose
        self.max_semitone_shift = 1.0  # Maximum ±1 půltón korekce

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inicializace komponent
        self.pitch_detector = AdvancedYINDetector()
        self.pitch_shifter = SimplePitchShifter()
        self.velocity_mapper = VelocityMapper()

    def generate_unique_filename(self, midi, velocity, sample_rate_str):
        """Generování unikátního názvu souboru s -next1, -next2, atd."""
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
        """Fáze 1: Načtení a analýza všech souborů"""
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
                # Zajištění 2D formátu (vzorky, kanály)
                if len(waveform.shape) == 1:
                    waveform = waveform[:, np.newaxis]
                print(f"[DEBUG] Úspěšně načten: {waveform.shape[0]} vzorků, {waveform.shape[1]} kanálů, {sr} Hz")
            except Exception as e:
                print(f"[ERROR] Chyba při načítání {file_path.name}: {e}")
                continue

            # Validace vzorkovací frekvence
            if sr not in [44100, 48000]:
                print(f"[WARNING] Nepodporovaná vzorkovací frekvence {sr} Hz pro {file_path.name}")
                continue

            # Výpočet délky (bez limitu)
            duration = len(waveform) / sr

            # Aktualizace parametrů pitch detektoru podle sample rate
            self.pitch_detector.update_parameters_for_sample_rate(sr)

            # Detekce pitch pomocí pokročilého YIN (převod na 1D pro detekci)
            waveform_1d = waveform[:, 0] if waveform.shape[1] > 1 else waveform.flatten()
            detected_pitch = self.pitch_detector.detect(waveform_1d, verbose=self.verbose)

            if detected_pitch is None:
                print(f"[WARNING] Nelze detekovat pitch pro {file_path.name}")
                continue

            midi_float = AudioUtils.freq_to_midi(detected_pitch)

            # Najdi nejbližší MIDI notu (ne jen zaokrouhlení)
            midi_lower = int(np.floor(midi_float))
            midi_upper = int(np.ceil(midi_float))

            # Výpočet vzdáleností k oběma notám
            dist_lower = abs(midi_float - midi_lower)
            dist_upper = abs(midi_float - midi_upper)

            # Vyber tu s menší korekcí
            if dist_lower <= dist_upper:
                midi_int = midi_lower
                semitone_diff = midi_float - midi_lower
            else:
                midi_int = midi_upper
                semitone_diff = midi_float - midi_upper

            # Kontrola, zda je korekce v přípustném rozsahu (±1 půltón)
            if abs(semitone_diff) > self.max_semitone_shift:
                print(
                    f"[WARNING] Soubor {file_path.name} vyžaduje korekci {semitone_diff:.2f} půltónů (>±{self.max_semitone_shift}), zahazuji")
                continue

            # Výpočet RMS
            rms_db = AudioUtils.calculate_rms_db(waveform)

            sample = AudioSample(file_path, waveform, sr, rms_db, midi_int)
            samples.append(sample)

            note_name = AudioUtils.midi_to_note_name(midi_int)
            target_freq = AudioUtils.midi_to_freq(midi_int)
            print(f"  Pitch: {detected_pitch:.2f} Hz → MIDI {midi_int} ({note_name}, {target_freq:.2f} Hz)")
            print(f"  Korekce: {semitone_diff:+.3f} půltónů")
            print(f"  RMS: {rms_db:.2f} dB, SR: {sr} Hz, délka: {duration:.2f}s")

        print(f"\nCelkem načteno {len(samples)} zpracovatelných vzorků")
        return samples

    def create_velocity_mappings(self, samples):
        """Fáze 2: Vytvoření velocity mappingů pro každou MIDI notu"""
        print("\n=== FÁZE 2: Tvorba velocity map ===")

        # Seskupení vzorků podle MIDI noty
        midi_groups = defaultdict(list)
        for sample in samples:
            midi_groups[sample.midi_note].append(sample)

        velocity_mappings = {}

        for midi_note, midi_samples in midi_groups.items():
            note_name = AudioUtils.midi_to_note_name(midi_note)
            rms_values = [s.rms_db for s in midi_samples]

            print(f"\nMIDI {midi_note} ({note_name}): {len(midi_samples)} vzorků")
            print(f"  RMS rozsah: {min(rms_values):.2f} až {max(rms_values):.2f} dB")

            # Odstranění outlierů
            filtered_rms = self.velocity_mapper.remove_outliers(rms_values, self.outlier_threshold)
            outliers_removed = len(rms_values) - len(filtered_rms)

            if outliers_removed > 0:
                print(f"  Odstraněno {outliers_removed} outlierů")

            # Vytvoření velocity mappingu
            velocity_map = self.velocity_mapper.create_velocity_mapping(filtered_rms)
            velocity_mappings[midi_note] = velocity_map

            # Přiřazení velocity ke vzorkům
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
        """Fáze 3: Zpracování a export souborů"""
        print("\n=== FÁZE 3: Tuning a export ===")

        total_outputs = 0

        for sample in samples:
            if sample.velocity is None:
                continue  # Přeskoč outliery

            midi_note = sample.midi_note
            note_name = AudioUtils.midi_to_note_name(midi_note)

            print(f"\nZpracovávám: {sample.filepath.name}")
            print(f"  MIDI {midi_note} ({note_name}) → velocity {sample.velocity}")

            # Pitch korekce
            target_freq = AudioUtils.midi_to_freq(midi_note)

            # Re-detekce pitch pro přesnou korekci
            self.pitch_detector.update_parameters_for_sample_rate(sample.sr)
            waveform_1d = sample.waveform[:, 0] if sample.waveform.shape[1] > 1 else sample.waveform.flatten()
            detected_pitch = self.pitch_detector.detect(waveform_1d, verbose=False)

            if detected_pitch:
                semitone_shift = 12 * np.log2(target_freq / detected_pitch)

                # Dvojitá kontrola limitu posunu
                if abs(semitone_shift) > self.max_semitone_shift:
                    print(f"  [WARNING] Korekce {semitone_shift:.3f} půltónů překračuje limit, přeskakuji")
                    continue

                print(
                    f"  Pitch korekce: {detected_pitch:.2f} Hz → {target_freq:.2f} Hz ({semitone_shift:+.3f} půltónů)")

                # Aplikace jednoduchého pitch shift (mění délku)
                tuned_waveform, tuned_sr = self.pitch_shifter.pitch_shift_simple(
                    sample.waveform, sample.sr, semitone_shift
                )

                # Výpočet nové délky
                original_duration = len(sample.waveform) / sample.sr
                new_duration = len(tuned_waveform) / tuned_sr if len(tuned_waveform.shape) == 1 else len(
                    tuned_waveform) / tuned_sr
                print(f"  Délka: {original_duration:.3f}s → {new_duration:.3f}s")

            else:
                tuned_waveform = sample.waveform
                tuned_sr = sample.sr
                print(f"  Bez pitch korekce (re-detection failed)")

            # Generování výstupů pro oba cílové sample rate
            target_sample_rates = [(44100, 'f44'), (48000, 'f48')]

            for target_sr, sr_suffix in target_sample_rates:
                # Konverze sample rate pokud je potřeba
                if tuned_sr != target_sr:
                    try:
                        if len(tuned_waveform.shape) > 1 and tuned_waveform.shape[1] > 1:
                            # Multi-channel audio
                            converted_channels = []
                            for ch in range(tuned_waveform.shape[1]):
                                converted = resampy.resample(tuned_waveform[:, ch], tuned_sr, target_sr)
                                converted_channels.append(converted)
                            output_waveform = np.column_stack(converted_channels)
                        else:
                            # Mono audio
                            waveform_flat = tuned_waveform.flatten() if len(
                                tuned_waveform.shape) > 1 else tuned_waveform
                            output_waveform = resampy.resample(waveform_flat, tuned_sr, target_sr)
                            output_waveform = output_waveform[:, np.newaxis]
                    except Exception as e:
                        print(f"  [ERROR] Chyba při konverzi sample rate: {e}")
                        continue
                else:
                    output_waveform = tuned_waveform

                # Generování unikátního názvu souboru
                output_path = self.generate_unique_filename(midi_note, sample.velocity, sr_suffix)
                print(f"[DEBUG] Připravuji zápis: {output_path}")

                # Uložení souboru
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
        """Hlavní pipeline zpracování"""
        print(f"Vstupní adresář: {self.input_dir}")
        print(f"Výstupní adresář: {self.output_dir}")
        print(f"Outlier threshold: {self.outlier_threshold} dB")
        print(f"Max pitch korekce: ±{self.max_semitone_shift} půltónů")
        print(f"Pokročilá YIN detekce s oktávovou korekcí a sustain analýzou")

        # Fáze 1: Načtení a analýza
        samples = self.load_and_analyze_files()
        if not samples:
            return

        # Fáze 2: Vytvoření velocity mappingů
        velocity_mappings = self.create_velocity_mappings(samples)

        # Fáze 3: Zpracování a export
        total_outputs = self.process_and_export(samples)

        print(f"\n=== DOKONČENO ===")
        print(f"Celkem vytvořeno {total_outputs} výstupních souborů")
        print(f"Výstupní adresář: {self.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Program pro korekci pitch a velocity mapping s pokročilou YIN detekcí.

        Klíčové funkce:
        - Pokročilá YIN pitch detekce s oktávovou korekcí pro extrémní frekvence
        - Sustain analýza pro stabilnější detekci
        - Adaptivní parametry podle vzorkovací frekvence (44.1kHz/48kHz)  
        - Velocity mapping (0-7) podle RMS hlasitosti pro každou MIDI notu
        - Jednoduchý pitch shift (mění délku vzorku)
        - Dual-rate export (44.1kHz + 48kHz)
        - Maximální korekce ±1 půltón

        Optimalizováno pro elektrické piano vzorky s dynamickým envelope.

        Příklad použití:
        python pitch_corrector.py --input-dir ./piano_samples --output-dir ./output --verbose
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
                        help='Podrobný výstup pro debugging pitch detekce')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=== PITCH CORRECTOR S POKROČILOU YIN DETEKCÍ ===")
    print("Optimalizováno pro elektrické piano vzorky")
    print("Verze s oktávovou korekcí a sustain analýzou")
    print("=" * 50)

    corrector = PitchCorrectorWithVelocityMapping(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        outlier_threshold=args.outlier_threshold,
        verbose=args.verbose
    )

    corrector.process_all()