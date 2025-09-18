"""
Program pro korekci pitch a velocity mapping vzorků hudebních nástrojů
s pokročilou YIN detekcí a adaptivní oktávovou korekcí.

Autor: Refaktorovaná verze pro obecné hudební nástroje
Datum: 2025
"""

import argparse
import soundfile as sf
import numpy as np
import resampy
from collections import defaultdict
import statistics
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import os

# --- Konfigurace logování ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Vypnutí progress bar při verbose režimu pro lepší čitelnost
class ProgressManager:
    """Správce progress barů a výstupu"""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def info(self, message):
        """Informační zpráva"""
        if self.verbose:
            print(f"[INFO] {message}")
        else:
            tqdm.write(f"[INFO] {message}")

    def debug(self, message):
        """Debug zpráva (pouze v verbose režimu)"""
        if self.verbose:
            print(f"[DEBUG] {message}")

    def warning(self, message):
        """Varovná zpráva"""
        if self.verbose:
            print(f"[WARNING] {message}")
        else:
            tqdm.write(f"[WARNING] {message}")

    def error(self, message):
        """Chybová zpráva"""
        if self.verbose:
            print(f"[ERROR] {message}")
        else:
            tqdm.write(f"[ERROR] {message}")

    def section(self, title):
        """Sekce - hlavička"""
        separator = "=" * len(title)
        if self.verbose:
            print(f"\n{separator}")
            print(title)
            print(separator)
        else:
            tqdm.write(f"\n{separator}")
            tqdm.write(title)
            tqdm.write(separator)

class AudioUtils:
    """Pomocné funkce pro práci s audio daty"""

    MIDI_TO_NOTE = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    @staticmethod
    def freq_to_midi(freq):
        """Převod frekvence na MIDI číslo"""
        if freq <= 0:
            raise ValueError("Frekvence musí být kladná")
        return 12 * np.log2(freq / 440.0) + 69

    @staticmethod
    def midi_to_freq(midi):
        """Převod MIDI čísla na frekvenci"""
        return 440.0 * 2 ** ((midi - 69) / 12)

    @staticmethod
    def midi_to_note_name(midi):
        """Převod MIDI čísla na název noty (např. 60 -> C4)"""
        if not (0 <= midi <= 127):
            raise ValueError(f"MIDI číslo {midi} je mimo povolený rozsah 0-127")

        octave = (midi // 12) - 1
        note_idx = midi % 12
        note = AudioUtils.MIDI_TO_NOTE[note_idx]
        if len(note) == 1:
            note += '_'  # Padding pro konzistentní formátování
        return f"{note}{octave}"

    @staticmethod
    def calculate_rms_db(waveform):
        """Výpočet RMS hlasitosti v dB"""
        # Flatten pro multi-channel audio
        if len(waveform.shape) > 1:
            waveform = waveform.flatten()

        rms = np.sqrt(np.mean(waveform ** 2))
        if rms == 0:
            return -np.inf
        return 20 * np.log10(rms)

    @staticmethod
    def normalize_audio(waveform, target_db=-20.0):
        """Normalizace audio na cílovou úroveň v dB"""
        # Flatten pro multi-channel audio při výpočtu RMS
        if len(waveform.shape) > 1:
            rms = np.sqrt(np.mean(waveform.flatten() ** 2))
        else:
            rms = np.sqrt(np.mean(waveform ** 2))

        if rms == 0:
            return waveform

        target_rms = 10 ** (target_db / 20)
        return waveform * (target_rms / rms)


class AdvancedYINDetector:
    """
    Pokročilý YIN pitch detector s adaptivními parametry a oktávovou korekcí.
    Optimalizován pro vzorky hudebních nástrojů s dynamickým envelope.
    """

    def __init__(self, sample_rate=44100, progress_mgr=None):
        self.sample_rate = sample_rate
        self.min_duration = 0.1
        self.progress_mgr = progress_mgr or ProgressManager()

        # Rozsah frekvencí pro hudební nástroje
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

        self.progress_mgr.debug(f"YIN parametry pro {sample_rate}Hz: frame={self.frame_length}, hop={self.hop_size}")

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

        self.progress_mgr.debug(
            f"Sustain detekce: {len(sustain_audio)}/{len(waveform)} vzorků ({len(sustain_audio) / len(waveform) * 100:.1f}%)")
        return sustain_audio

    def preprocess_for_octave(self, waveform, rough_pitch):
        """
        Pre-processing pro oktávovou korekci podle hrubé pitch detekce.
        Posune extrémní frekvence do optimálního detekčního pásma.
        """
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
                try:
                    processed_audio = resampy.resample(waveform, self.sample_rate, new_sr)
                except Exception as e:
                    self.progress_mgr.error(f"Chyba při downsampling: {e}")
                    return waveform, self.sample_rate, 0, None

                self.progress_mgr.debug(
                    f"Vysoký tón korekce: {rough_pitch:.1f}Hz -> shift -{octave_shift} oktáv, nový SR: {new_sr}Hz")
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
                try:
                    processed_audio = resampy.resample(waveform, self.sample_rate, new_sr)
                except Exception as e:
                    self.progress_mgr.error(f"Chyba při upsampling: {e}")
                    return waveform, self.sample_rate, 0, None

                self.progress_mgr.debug(
                    f"Nízký tón korekce: {rough_pitch:.1f}Hz -> shift +{octave_shift} oktáv, nový SR: {new_sr}Hz")
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
                available_length = min(len(audio) - tau, len(audio))
                if available_length <= 0:
                    diff[tau] = np.inf
                else:
                    diff[tau] = np.sum((audio[:available_length] - audio[tau:tau + available_length]) ** 2)
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
        if len(audio) < 10:  # Minimální délka pro analýzu
            return None

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
            self.progress_mgr.debug(f"Chyba v basic YIN detekci: {e}")
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
                self.progress_mgr.debug(f"Signál je příliš krátký ({duration:.3f}s)")
            return None

        if verbose:
            self.progress_mgr.debug(f"Začínám pokročilou YIN detekci pro {duration:.2f}s audio")

        # 1. Základní pre-processing
        audio = self.preprocess_audio(waveform)

        # 2. Nalezení sustain sekce pro stabilnější detekci
        sustain_audio = self.find_sustain_section(audio)

        # 3. Hrubá detekce pro určení oktávové korekce
        rough_pitch = self.basic_yin_detection(sustain_audio, self.sample_rate)

        if rough_pitch is None:
            if verbose:
                self.progress_mgr.debug("Hrubá detekce selhala")
            return None

        if verbose:
            self.progress_mgr.debug(f"Hrubá detekce: {rough_pitch:.2f} Hz")

        # 4. Adaptivní oktávové předzpracování
        processed_audio, effective_sr, octave_shift, shift_direction = self.preprocess_for_octave(
            sustain_audio, rough_pitch
        )

        # 5. Finální přesná detekce
        final_pitch = self.basic_yin_detection(processed_audio, effective_sr)

        if final_pitch is None:
            if verbose:
                self.progress_mgr.debug("Finální detekce selhala, používám hrubou detekci")
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
            self.progress_mgr.debug(f"Finální pitch: {corrected_pitch:.2f} Hz (korekce: {octave_shift} oktáv)")

        # Validace finálního výsledku
        if self.fmin <= corrected_pitch <= self.fmax:
            return corrected_pitch
        else:
            if verbose:
                self.progress_mgr.debug(f"Korigovaný pitch {corrected_pitch:.2f} Hz je mimo rozsah, používám hrubou detekci")
            return rough_pitch


class SimplePitchShifter:
    """Jednoduchý pitch shifting pomocí resample (mění délku)"""

    def __init__(self, progress_mgr=None):
        self.progress_mgr = progress_mgr or ProgressManager()

    def pitch_shift_simple(self, audio, sr, semitone_shift):
        """
        Pitch shift změnou sample rate - mění délku vzorku.
        Vhodné pro vzorky kde kratší vysoké/delší nízké tóny jsou realistické.
        """
        if abs(semitone_shift) < 0.01:
            return audio, sr

        factor = 2 ** (semitone_shift / 12)
        new_sr = int(sr * factor)

        try:
            # Zpracování multi-channel audio
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                shifted_channels = []
                for ch in range(audio.shape[1]):
                    shifted = resampy.resample(audio[:, ch], sr, new_sr)
                    shifted_channels.append(shifted)
                shifted_audio = np.column_stack(shifted_channels)
            else:
                # Mono nebo 2D s jedním kanálem
                audio_1d = audio.flatten() if len(audio.shape) > 1 else audio
                shifted_audio = resampy.resample(audio_1d, sr, new_sr)

                # Zachovat originální formát
                if len(audio.shape) > 1:
                    shifted_audio = shifted_audio[:, np.newaxis]

            return shifted_audio, new_sr

        except Exception as e:
            self.progress_mgr.error(f"Chyba při pitch shift: {e}")
            return audio, sr


class VelocityMapper:
    """Správa velocity mappingu na základě seskupování podobných RMS hodnot"""

    @staticmethod
    def create_velocity_mapping(rms_values, grouping_threshold=1.5):
        """
        Vytvoření velocity mappingu (0-7) z RMS hodnot na základě seskupování.
        - Seřadí RMS hodnoty vzestupně.
        - Vytváří skupiny s max rozdílem 2 * grouping_threshold (např. 3 dB).
        - Přiřadí velocity 0 (nejtišší) až 7 (nejsilnější), max 8 skupin.
        - Pokud by skupin bylo více než 8, vyřadí extrémní vzorky.
        - Vzorky mimo skupiny dostanou None.
        """
        if len(rms_values) <= 1:
            return {0: list(rms_values)}

        # Seřazení RMS hodnot vzestupně (od nejtišší po nejsilnější)
        sorted_rms = sorted(rms_values)

        velocity_map = defaultdict(list)
        current_group_min = sorted_rms[0]
        current_velocity = 0

        for rms in sorted_rms:
            if abs(rms - current_group_min) <= 2 * grouping_threshold:
                # Vzorek spadá do aktuální skupiny
                velocity_map[current_velocity].append(rms)
            else:
                # Vytvoření nové skupiny, pokud je méně než 8
                if current_velocity < 7:
                    current_velocity += 1
                    current_group_min = rms
                    velocity_map[current_velocity].append(rms)
                else:
                    # Překročen limit 8 skupin - vyřadit vzorek (None)
                    pass  # Extrémní vzorek se nezařadí

        # Pokud skupin je méně než 8, je to OK
        return dict(velocity_map)


class AudioSample:
    """Container pro data audio vzorku"""

    def __init__(self, filepath, waveform, sr, rms_db, midi_note, detected_pitch=None):
        self.filepath = filepath
        self.waveform = waveform
        self.sr = sr
        self.rms_db = rms_db
        self.midi_note = midi_note
        self.detected_pitch = detected_pitch  # Uložená detekovaná pitch pro opětovné použití
        self.velocity = None

        # Validace dat
        if not isinstance(filepath, Path):
            self.filepath = Path(filepath)
        if not (0 <= midi_note <= 127):
            raise ValueError(f"MIDI nota {midi_note} je mimo rozsah 0-127")


class PitchCorrectorWithVelocityMapping:
    """
    Hlavní třída procesoru pro korekci pitch a velocity mapping.
    Optimalizována pro vzorky hudebních nástrojů.
    """

    def __init__(self, input_dir, output_dir, grouping_threshold=1.5, verbose=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.grouping_threshold = grouping_threshold
        self.verbose = verbose
        self.max_semitone_shift = 1.0  # Maximum ±1 půltón korekce

        # Validace adresářů
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Vstupní adresář neexistuje: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inicializace progress manageru a komponent
        self.progress_mgr = ProgressManager(verbose=verbose)
        self.pitch_detector = AdvancedYINDetector(progress_mgr=self.progress_mgr)
        self.pitch_shifter = SimplePitchShifter(progress_mgr=self.progress_mgr)
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
        self.progress_mgr.section("FÁZE 1: Načítání a analýza souborů")
        samples = []

        # Nalezení WAV souborů
        wav_files = list(self.input_dir.glob("*.wav")) + list(self.input_dir.glob("*.WAV"))

        if not wav_files:
            self.progress_mgr.error("Nebyly nalezeny žádné WAV soubory!")
            return samples

        self.progress_mgr.info(f"Nalezeno {len(wav_files)} WAV souborů")

        # Progress bar pro načítání souborů (vypnutý v verbose režimu)
        iterator = wav_files if self.verbose else tqdm(wav_files, desc="Načítám soubory", unit="soubor")

        for file_path in iterator:
            if self.verbose:
                print(f"\n--- Zpracovávám: {file_path.name} ---")
            else:
                # Odřádkování před informacemi o souboru
                tqdm.write(f"\nZpracovávám: {file_path.name}")

            self.progress_mgr.debug(f"Čtu soubor: {file_path}")

            try:
                waveform, sr = sf.read(str(file_path))
                # Zajištění 2D formátu (vzorky, kanály)
                if len(waveform.shape) == 1:
                    waveform = waveform[:, np.newaxis]
                self.progress_mgr.debug(f"Načten: {waveform.shape[0]} vzorků, {waveform.shape[1]} kanálů, {sr} Hz")
            except Exception as e:
                self.progress_mgr.error(f"Chyba při načítání {file_path.name}: {e}")
                continue

            # Validace vzorkovací frekvence
            if sr not in [44100, 48000]:
                self.progress_mgr.warning(f"Nepodporovaná vzorkovací frekvence {sr} Hz pro {file_path.name}")
                continue

            # Výpočet délky
            duration = len(waveform) / sr

            # Aktualizace parametrů pitch detektoru podle sample rate
            self.pitch_detector.update_parameters_for_sample_rate(sr)

            # Detekce pitch pomocí pokročilého YIN (převod na 1D pro detekci)
            waveform_1d = waveform[:, 0] if waveform.shape[1] > 1 else waveform.flatten()
            detected_pitch = self.pitch_detector.detect(waveform_1d, verbose=self.verbose)

            if detected_pitch is None:
                self.progress_mgr.warning(f"Nelze detekovat pitch pro {file_path.name}")
                continue

            try:
                midi_float = AudioUtils.freq_to_midi(detected_pitch)
            except ValueError as e:
                self.progress_mgr.error(f"Chyba při převodu frekvence na MIDI pro {file_path.name}: {e}")
                continue

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

            # Kontrola rozsahu MIDI
            if not (0 <= midi_int <= 127):
                self.progress_mgr.warning(f"MIDI nota {midi_int} mimo rozsah 0-127 pro {file_path.name}")
                continue

            # Kontrola, zda je korekce v přípustném rozsahu
            if abs(semitone_diff) > self.max_semitone_shift:
                self.progress_mgr.warning(
                    f"Soubor {file_path.name} vyžaduje korekci {semitone_diff:.2f} půltónů (>±{self.max_semitone_shift}), zahazuji")
                continue

            # Výpočet RMS
            rms_db = AudioUtils.calculate_rms_db(waveform)
            if rms_db == -np.inf:
                self.progress_mgr.warning(f"Soubor {file_path.name} obsahuje pouze ticho, přeskakuji")
                continue

            try:
                sample = AudioSample(file_path, waveform, sr, rms_db, midi_int, detected_pitch=detected_pitch)
                samples.append(sample)
            except ValueError as e:
                self.progress_mgr.error(f"Chyba při vytváření vzorku pro {file_path.name}: {e}")
                continue

            note_name = AudioUtils.midi_to_note_name(midi_int)
            target_freq = AudioUtils.midi_to_freq(midi_int)

            # Informace o zpracovaném souboru
            info_lines = [
                f"  Pitch: {detected_pitch:.2f} Hz → MIDI {midi_int} ({note_name}, {target_freq:.2f} Hz)",
                f"  Korekce: {semitone_diff:+.3f} půltónů",
                f"  RMS: {rms_db:.2f} dB, SR: {sr} Hz, délka: {duration:.2f}s"
            ]

            for line in info_lines:
                if self.verbose:
                    print(line)
                else:
                    tqdm.write(line)

        result_msg = f"Celkem načteno {len(samples)} zpracovatelných vzorků"
        if self.verbose:
            print(f"\n{result_msg}")
        else:
            tqdm.write(f"\n{result_msg}")

        return samples

    def create_velocity_mappings(self, samples):
        """Fáze 2: Vytvoření velocity mappingů pro každou MIDI notu"""
        self.progress_mgr.section("FÁZE 2: Tvorba velocity map")

        # Seskupení vzorků podle MIDI noty
        midi_groups = defaultdict(list)
        for sample in samples:
            midi_groups[sample.midi_note].append(sample)

        velocity_mappings = {}

        for midi_note, midi_samples in midi_groups.items():
            note_name = AudioUtils.midi_to_note_name(midi_note)
            rms_values = [s.rms_db for s in midi_samples]

            info_lines = [
                f"\nMIDI {midi_note} ({note_name}): {len(midi_samples)} vzorků",
                f"  RMS rozsah: {min(rms_values):.2f} až {max(rms_values):.2f} dB"
            ]

            for line in info_lines:
                if self.verbose:
                    print(line)
                else:
                    tqdm.write(line)

            # Vytvoření velocity mappingu na základě seskupování
            velocity_map = self.velocity_mapper.create_velocity_mapping(rms_values, self.grouping_threshold)
            velocity_mappings[midi_note] = velocity_map

            # Přiřazení velocity ke vzorkům
            for sample in midi_samples:
                sample.velocity = None
                for vel, rms_list in velocity_map.items():
                    if sample.rms_db in rms_list:
                        sample.velocity = vel
                        break

            # Zobrazení velocity mappingu
            mapping_header = "  Velocity mapping:"
            if self.verbose:
                print(mapping_header)
            else:
                tqdm.write(mapping_header)

            for vel, rms_list in sorted(velocity_map.items()):
                mapping_line = f"    vel{vel}: {len(rms_list)} vzorků (RMS {min(rms_list):.1f} až {max(rms_list):.1f} dB)"
                if self.verbose:
                    print(mapping_line)
                else:
                    tqdm.write(mapping_line)

        return velocity_mappings

    def process_and_export(self, samples):
        """Fáze 3: Zpracování a export souborů"""
        self.progress_mgr.section("FÁZE 3: Tuning a export")

        total_outputs = 0
        valid_samples = [s for s in samples if s.velocity is not None]

        if not valid_samples:
            self.progress_mgr.error("Žádné vzorky k exportu!")
            return 0

        self.progress_mgr.info(f"Exportuji {len(valid_samples)} vzorků")

        # Progress bar pro zpracování vzorků (vypnutý v verbose režimu)
        iterator = valid_samples if self.verbose else tqdm(valid_samples, desc="Exportuji vzorky", unit="vzorek")

        for sample in iterator:
            midi_note = sample.midi_note
            note_name = AudioUtils.midi_to_note_name(midi_note)

            if self.verbose:
                print(f"\n--- Exportuji: {sample.filepath.name} ---")
                print(f"MIDI {midi_note} ({note_name}) → velocity {sample.velocity}")
            else:
                tqdm.write(f"\nExportuji: {sample.filepath.name}")
                tqdm.write(f"  MIDI {midi_note} ({note_name}) → velocity {sample.velocity}")

            # Pitch korekce
            target_freq = AudioUtils.midi_to_freq(midi_note)
            detected_pitch = sample.detected_pitch  # Použití uložené hodnoty

            if detected_pitch:
                semitone_shift = 12 * np.log2(target_freq / detected_pitch)

                # Dvojitá kontrola limitu posunu
                if abs(semitone_shift) > self.max_semitone_shift:
                    self.progress_mgr.warning(f"Korekce {semitone_shift:.3f} půltónů překračuje limit, přeskakuji")
                    continue

                correction_info = f"  Pitch korekce: {detected_pitch:.2f} Hz → {target_freq:.2f} Hz ({semitone_shift:+.3f} půltónů)"
                if self.verbose:
                    print(correction_info)
                else:
                    tqdm.write(correction_info)

                # Aplikace jednoduchého pitch shift (mění délku)
                tuned_waveform, tuned_sr = self.pitch_shifter.pitch_shift_simple(
                    sample.waveform, sample.sr, semitone_shift
                )

                # Výpočet nové délky
                original_duration = len(sample.waveform) / sample.sr
                new_duration = len(tuned_waveform) / tuned_sr
                duration_info = f"  Délka: {original_duration:.3f}s → {new_duration:.3f}s"
                if self.verbose:
                    print(duration_info)
                else:
                    tqdm.write(duration_info)

            else:
                tuned_waveform = sample.waveform
                tuned_sr = sample.sr
                no_correction_info = "  Bez pitch korekce (detekce selhala)"
                if self.verbose:
                    print(no_correction_info)
                else:
                    tqdm.write(no_correction_info)

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
                            # Mono audio nebo 2D s jedním kanálem
                            waveform_flat = tuned_waveform.flatten() if len(
                                tuned_waveform.shape) > 1 else tuned_waveform
                            output_waveform = resampy.resample(waveform_flat, tuned_sr, target_sr)
                            # Zachovat 2D formát
                            output_waveform = output_waveform[:, np.newaxis]
                    except Exception as e:
                        self.progress_mgr.error(f"Chyba při konverzi sample rate: {e}")
                        continue
                else:
                    output_waveform = tuned_waveform

                # Generování unikátního názvu souboru
                output_path = self.generate_unique_filename(midi_note, sample.velocity, sr_suffix)
                self.progress_mgr.debug(f"Připravuji zápis: {output_path}")

                # Uložení souboru
                try:
                    sf.write(str(output_path), output_waveform, target_sr)
                    save_info = f"  Uložen: {output_path.name}"
                    if self.verbose:
                        print(save_info)
                    else:
                        tqdm.write(save_info)
                    self.progress_mgr.debug(f"Úspěšně zapsán: {output_path} ({len(output_waveform)} vzorků, {target_sr} Hz)")
                    total_outputs += 1
                except Exception as e:
                    self.progress_mgr.error(f"Chyba při ukládání {output_path.name}: {e}")
                    self.progress_mgr.debug(f"Neúspěšný zápis do: {output_path}")

        return total_outputs

    def process_all(self):
        """Hlavní pipeline zpracování"""
        print(f"Vstupní adresář: {self.input_dir}")
        print(f"Výstupní adresář: {self.output_dir}")
        print(f"Grouping threshold: {self.grouping_threshold} dB")
        print(f"Max pitch korekce: ±{self.max_semitone_shift} půltónů")
        print(f"Pokročilá YIN detekce s oktávovou korekcí a sustain analýzou")
        print(f"Verbose režim: {'ZAPNUT' if self.verbose else 'VYPNUT'}")

        try:
            # Fáze 1: Načtení a analýza
            samples = self.load_and_analyze_files()
            if not samples:
                return

            # Fáze 2: Vytvoření velocity mappingů
            velocity_mappings = self.create_velocity_mappings(samples)

            # Fáze 3: Zpracování a export
            total_outputs = self.process_and_export(samples)

            # Finální shrnutí
            self.progress_mgr.section("DOKONČENO")
            summary_lines = [
                f"Celkem vytvořeno {total_outputs} výstupních souborů",
                f"Výstupní adresář: {self.output_dir}"
            ]

            for line in summary_lines:
                if self.verbose:
                    print(line)
                else:
                    tqdm.write(line)

        except KeyboardInterrupt:
            self.progress_mgr.error("Zpracování přerušeno uživatelem")
        except Exception as e:
            self.progress_mgr.error(f"Neočekávaná chyba: {e}")
            raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="""Program pro korekci pitch a velocity mapping s pokročilou YIN detekcí.

        Klíčové funkce:
        - Pokročilá YIN pitch detekce s oktávovou korekcí pro extrémní frekvence
        - Sustain analýza pro stabilnější detekci
        - Adaptivní parametry podle vzorkovací frekvence (44.1kHz/48kHz)  
        - Velocity mapping (0-7) podle RMS hlasitosti pro každou MIDI notu na základě seskupování podobných hodnot
        - Jednoduchý pitch shift (mění délku vzorku)
        - Dual-rate export (44.1kHz + 48kHz)
        - Maximální korekce ±1 půltón
        - Vylepšené informování o progresu s podporou verbose režimu

        Optimalizováno pro vzorky hudebních nástrojů s dynamickým envelope.

        Příklad použití:
        python pitch_corrector.py --input-dir ./samples --output-dir ./output --verbose
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--input-dir', required=True,
                        help='Cesta k adresáři s vstupními WAV soubory')
    parser.add_argument('--output-dir', required=True,
                        help='Cesta k výstupnímu adresáři')
    parser.add_argument('--grouping-threshold', type=float, default=1.5,
                        help='Práh pro seskupování RMS hodnot v dB (výchozí: 1.5)')
    parser.add_argument('--verbose', action='store_true',
                        help='Podrobný výstup pro debugging (vypne progress bary pro lepší čitelnost)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=== PITCH CORRECTOR S POKROČILOU YIN DETEKCÍ ===")
    print("Optimalizováno pro vzorky hudebních nástrojů")
    print("Verze s oktávovou korekcí a sustain analýzou")
    print("=" * 50)

    try:
        corrector = PitchCorrectorWithVelocityMapping(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            grouping_threshold=args.grouping_threshold,
            verbose=args.verbose
        )

        corrector.process_all()

    except FileNotFoundError as e:
        print(f"CHYBA: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nZpracování přerušeno uživatelem")
        sys.exit(1)
    except Exception as e:
        print(f"KRITICKÁ CHYBA: {e}")
        sys.exit(1)