"""
Refaktorovaný program pro korekci pitch a velocity mapping
s vylepšenou harmonickou detekcí a robustní architekturou.

Klíčové vylepšení:
- Modernní harmonická detekce místo problematické oktávové korekce
- Spektrální analýza kombinovaná s YIN algoritmem
- Robustní velocity mapping založený na multiple peak metrics
- Modulární architektura s jasným separation of concerns
- Comprehensive error handling a logging
"""

import argparse
import soundfile as sf
import numpy as np
import resampy
from collections import defaultdict
from pathlib import Path
import logging
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sys

# Konfigurace loggingu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PitchAnalysisResult:
    """Výsledek pitch analýzy s detailními metrikami"""
    fundamental_freq: Optional[float]
    confidence: float
    harmonics: List[float]
    method_used: str
    spectral_centroid: float
    spectral_rolloff: float

    @property
    def midi_note(self) -> Optional[int]:
        if self.fundamental_freq is None:
            return None
        return AudioUtils.freq_to_midi(self.fundamental_freq)


@dataclass
class VelocityAnalysisResult:
    """Výsledek velocity analýzy"""
    rms_db: float
    peak_db: float
    attack_peak_db: float
    attack_rms_db: float
    dynamic_range: float


@dataclass
class AudioSampleData:
    """Kompletní data audio vzorku"""
    filepath: Path
    waveform: np.ndarray
    sample_rate: int
    duration: float
    pitch_analysis: Optional[PitchAnalysisResult]
    velocity_analysis: Optional[VelocityAnalysisResult]
    assigned_velocity: Optional[int] = None
    target_midi_note: Optional[int] = None
    pitch_correction_semitones: Optional[float] = None


class AudioUtils:
    """Rozšířené audio utility funkce"""

    MIDI_TO_NOTE = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    @staticmethod
    def freq_to_midi(freq: float) -> int:
        """Převod frekvence na MIDI číslo s validací"""
        if freq <= 0:
            raise ValueError("Frekvence musí být kladná")
        midi_float = 12 * np.log2(freq / 440.0) + 69
        return int(np.round(midi_float))

    @staticmethod
    def midi_to_freq(midi: int) -> float:
        """Převod MIDI čísla na frekvenci"""
        return 440.0 * 2 ** ((midi - 69) / 12)

    @staticmethod
    def midi_to_note_name(midi: int) -> str:
        """Převod MIDI čísla na název noty"""
        if not (0 <= midi <= 127):
            raise ValueError(f"MIDI číslo {midi} je mimo rozsah 0-127")

        octave = (midi // 12) - 1
        note_idx = midi % 12
        note = AudioUtils.MIDI_TO_NOTE[note_idx]
        return f"{note}{octave}"

    @staticmethod
    def normalize_audio(waveform: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """Normalizace audio na cílovou úroveň"""
        if len(waveform.shape) > 1:
            rms = np.sqrt(np.mean(waveform.flatten() ** 2))
        else:
            rms = np.sqrt(np.mean(waveform ** 2))

        if rms == 0:
            return waveform

        target_rms = 10 ** (target_db / 20)
        return waveform * (target_rms / rms)

    @staticmethod
    def to_mono(waveform: np.ndarray) -> np.ndarray:
        """Převod na mono s zachováním 1D formátu"""
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            return np.mean(waveform, axis=1)
        elif len(waveform.shape) > 1:
            return waveform[:, 0]
        return waveform

    @staticmethod
    def spectral_features(waveform: np.ndarray, sr: int) -> Tuple[float, float]:
        """Výpočet spektrálních charakteristik"""
        # FFT pro spektrální analýzu
        n_fft = min(2048, len(waveform))
        spectrum = np.abs(np.fft.fft(waveform[:n_fft]))
        freqs = np.fft.fftfreq(n_fft, 1/sr)[:n_fft//2]
        spectrum = spectrum[:n_fft//2]

        # Spektrální centroid
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0

        # Spektrální rolloff (85% energie)
        cumsum_spectrum = np.cumsum(spectrum)
        rolloff_threshold = 0.85 * cumsum_spectrum[-1]
        rolloff_idx = np.where(cumsum_spectrum >= rolloff_threshold)[0]
        rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else sr/2

        return centroid, rolloff


class PitchDetector(ABC):
    """Abstraktní base class pro pitch detektory"""

    @abstractmethod
    def detect(self, waveform: np.ndarray, sr: int) -> PitchAnalysisResult:
        pass


class HybridPitchDetector(PitchDetector):
    """
    Hybridní pitch detector kombinující spektrální analýzu s YIN algoritmem
    a robustní harmonickou filtrací
    """

    def __init__(self, fmin: float = 27.5, fmax: float = 4186.0):
        self.fmin = fmin
        self.fmax = fmax
        self.frame_length = 4096
        self.hop_length = 512

    def detect(self, waveform: np.ndarray, sr: int) -> PitchAnalysisResult:
        """Hlavní detekční metoda"""

        # Preprocessing
        audio = self._preprocess(waveform)

        # 1. Spektrální analýza pro kandidáty
        spectral_candidates = self._spectral_peak_detection(audio, sr)

        # 2. YIN analýza pro potvrzení
        yin_candidates = self._yin_analysis(audio, sr)

        # 3. Kombinace výsledků
        fundamental_freq = self._combine_candidates(spectral_candidates, yin_candidates)

        # 4. Harmonická analýza
        harmonics = self._detect_harmonics(audio, sr, fundamental_freq) if fundamental_freq else []

        # 5. Confidence score
        confidence = self._calculate_confidence(fundamental_freq, harmonics, audio, sr)

        # 6. Spektrální charakteristiky
        centroid, rolloff = AudioUtils.spectral_features(audio, sr)

        return PitchAnalysisResult(
            fundamental_freq=fundamental_freq,
            confidence=confidence,
            harmonics=harmonics,
            method_used="hybrid_spectral_yin",
            spectral_centroid=centroid,
            spectral_rolloff=rolloff
        )

    def _preprocess(self, waveform: np.ndarray) -> np.ndarray:
        """Vylepšené preprocessing s filtráciou"""
        audio = AudioUtils.to_mono(waveform)
        audio = AudioUtils.normalize_audio(audio, target_db=-20.0)

        # Aplikace filtrů pro lepší pitch detekci
        audio = self._apply_filters(audio)

        return audio

    def _apply_filters(self, audio: np.ndarray) -> np.ndarray:
        """Aplikace filtrů optimalizovaných pro pitch detekci"""

        # 1. High-pass filter - odstranění DC a velmi nízkých frekvencí
        audio = self._highpass_filter(audio, cutoff_hz=40.0)

        # 2. Notch filter pro síťové brumání (50/60 Hz)
        audio = self._notch_filter(audio, freq=50.0, quality=10.0)
        audio = self._notch_filter(audio, freq=60.0, quality=10.0)

        # 3. Gentle low-pass pro redukci vysokofrekvenčního šumu
        audio = self._lowpass_filter(audio, cutoff_hz=8000.0)

        return audio

    def _highpass_filter(self, audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """Jednoduchý high-pass filter pomocí rozdílové rovnice"""
        if len(audio) < 10:
            return audio

        # RC high-pass filter s cutoff frekvencí
        # Aproximace pro sample rate 44100-48000 Hz
        dt = 1.0 / 44100  # Aproximace
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        alpha = rc / (rc + dt)

        # Aplikace filtru
        filtered = np.zeros_like(audio)
        filtered[0] = audio[0]

        for i in range(1, len(audio)):
            filtered[i] = alpha * (filtered[i-1] + audio[i] - audio[i-1])

        return filtered

    def _lowpass_filter(self, audio: np.ndarray, cutoff_hz: float) -> np.ndarray:
        """Jednoduchý low-pass filter"""
        if len(audio) < 10:
            return audio

        # RC low-pass filter
        dt = 1.0 / 44100  # Aproximace
        rc = 1.0 / (2 * np.pi * cutoff_hz)
        alpha = dt / (rc + dt)

        # Aplikace filtru
        filtered = np.zeros_like(audio)
        filtered[0] = audio[0]

        for i in range(1, len(audio)):
            filtered[i] = filtered[i-1] + alpha * (audio[i] - filtered[i-1])

        return filtered

    def _notch_filter(self, audio: np.ndarray, freq: float, quality: float = 10.0) -> np.ndarray:
        """Jednoduchý notch filter pro odstranění konkrétní frekvence"""
        if len(audio) < 10:
            return audio

        # Zjednodušená implementace notch filtru
        # Pro produkční verzi by bylo lepší použít scipy.signal

        # Vytvoř sinusovou vlnu na cílové frekvenci
        sr = 44100  # Aproximace
        t = np.arange(len(audio)) / sr

        # Detekce amplitudy na cílové frekvenci pomocí korelace
        test_sin = np.sin(2 * np.pi * freq * t)
        test_cos = np.cos(2 * np.pi * freq * t)

        # Projekce signálu na sin/cos komponenty
        sin_coeff = np.dot(audio, test_sin) / len(audio)
        cos_coeff = np.dot(audio, test_cos) / len(audio)

        # Rekonstrukce komponenty na cílové frekvenci
        target_component = sin_coeff * test_sin + cos_coeff * test_cos

        # Odečtení (s faktorem útlumu)
        attenuation = 1.0 / quality  # Vyšší Q = větší útlum
        filtered = audio - attenuation * target_component

        return filtered

    def _spectral_peak_detection(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Detekce spektrálních peaků jako kandidátů na základní frekvenci"""
        candidates = []

        # Windowed analýza
        for start in range(0, len(audio) - self.frame_length, self.hop_length):
            frame = audio[start:start + self.frame_length]

            # Windowing function
            window = np.hanning(len(frame))
            windowed_frame = frame * window

            # FFT
            spectrum = np.abs(np.fft.fft(windowed_frame))
            freqs = np.fft.fftfreq(len(windowed_frame), 1/sr)

            # Pouze pozitivní frekvence
            half_spectrum = spectrum[:len(spectrum)//2]
            half_freqs = freqs[:len(freqs)//2]

            # Najdi lokální maxima
            peaks = self._find_spectral_peaks(half_spectrum, half_freqs)
            candidates.extend(peaks)

        # Seskupení podobných frekvencí
        return self._group_frequency_candidates(candidates)

    def _find_spectral_peaks(self, spectrum: np.ndarray, freqs: np.ndarray) -> List[Tuple[float, float]]:
        """Najde lokální maxima ve spektru"""
        peaks = []

        # Minimum peak height (relativně k max spektra)
        min_peak_height = 0.1 * np.max(spectrum)

        for i in range(2, len(spectrum) - 2):
            if (spectrum[i] > spectrum[i-1] and
                spectrum[i] > spectrum[i+1] and
                spectrum[i] > spectrum[i-2] and
                spectrum[i] > spectrum[i+2] and
                spectrum[i] > min_peak_height and
                self.fmin <= freqs[i] <= self.fmax):

                peaks.append((freqs[i], spectrum[i]))

        return peaks

    def _yin_analysis(self, audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """YIN analýza pro potvrzení kandidátů"""
        candidates = []

        for start in range(0, len(audio) - self.frame_length, self.hop_length):
            frame = audio[start:start + self.frame_length]
            pitch = self._yin_single_frame(frame, sr)

            if pitch and self.fmin <= pitch <= self.fmax:
                # Confidence based on YIN threshold
                confidence = 0.8  # Placeholder - v reálné implementaci by se počítala z YIN CMNDF
                candidates.append((pitch, confidence))

        return self._group_frequency_candidates(candidates)

    def _yin_single_frame(self, frame: np.ndarray, sr: int) -> Optional[float]:
        """Jednoduchá YIN implementace pro jeden frame"""
        if len(frame) < 100:
            return None

        # YIN parameters
        max_tau = min(len(frame) // 2, int(sr / self.fmin))
        min_tau = max(1, int(sr / self.fmax))

        if max_tau <= min_tau:
            return None

        # Difference function
        diff = np.zeros(max_tau)
        for tau in range(1, max_tau):
            available_length = len(frame) - tau
            if available_length > 0:
                diff[tau] = np.sum((frame[:-tau] - frame[tau:]) ** 2)

        # Cumulative mean normalized difference
        cmndf = np.zeros(max_tau)
        cmndf[0] = 1.0

        for tau in range(1, max_tau):
            if tau == 1:
                cmndf[tau] = diff[tau]
            else:
                mean_diff = np.sum(diff[1:tau+1]) / tau
                if mean_diff > 0:
                    cmndf[tau] = diff[tau] / mean_diff
                else:
                    cmndf[tau] = 1.0

        # Najdi první minimum pod prahem
        threshold = 0.3
        for tau in range(min_tau, max_tau):
            if cmndf[tau] < threshold:
                return sr / tau

        return None

    def _group_frequency_candidates(self, candidates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Seskupí podobné frekvence a spočítá průměry"""
        if not candidates:
            return []

        # Seřaď podle frekvence
        candidates.sort(key=lambda x: x[0])

        grouped = []
        current_group = [candidates[0]]

        for freq, confidence in candidates[1:]:
            # Tolerance 5% pro seskupení
            if abs(freq - current_group[-1][0]) / current_group[-1][0] < 0.05:
                current_group.append((freq, confidence))
            else:
                # Uzavři aktuální skupinu
                if current_group:
                    avg_freq = np.mean([f for f, c in current_group])
                    total_confidence = np.sum([c for f, c in current_group])
                    grouped.append((avg_freq, total_confidence))
                current_group = [(freq, confidence)]

        # Poslední skupina
        if current_group:
            avg_freq = np.mean([f for f, c in current_group])
            total_confidence = np.sum([c for f, c in current_group])
            grouped.append((avg_freq, total_confidence))

        return grouped

    def _combine_candidates(self, spectral_candidates: List[Tuple[float, float]],
                          yin_candidates: List[Tuple[float, float]]) -> Optional[float]:
        """Kombinuje kandidáty ze spektrální a YIN analýzy"""

        all_candidates = []

        # Spektrální kandidáti s váhou 0.6
        for freq, confidence in spectral_candidates:
            all_candidates.append((freq, confidence * 0.6, "spectral"))

        # YIN kandidáti s váhou 0.8 (YIN je spolehlivější pro fundamentální frekvenci)
        for freq, confidence in yin_candidates:
            all_candidates.append((freq, confidence * 0.8, "yin"))

        if not all_candidates:
            return None

        # Seskupení velmi podobných kandidátů z obou metod
        final_candidates = []

        for freq, confidence, method in all_candidates:
            # Najdi podobné kandidáty
            similar_found = False
            for i, (existing_freq, existing_conf, existing_methods) in enumerate(final_candidates):
                if abs(freq - existing_freq) / existing_freq < 0.03:  # 3% tolerance
                    # Aktualizuj existující kandidát
                    new_freq = (existing_freq * existing_conf + freq * confidence) / (existing_conf + confidence)
                    new_conf = existing_conf + confidence
                    new_methods = existing_methods + [method]
                    final_candidates[i] = (new_freq, new_conf, new_methods)
                    similar_found = True
                    break

            if not similar_found:
                final_candidates.append((freq, confidence, [method]))

        if not final_candidates:
            return None

        # Vyber kandidát s nejvyšší confidence
        # Bonus pro kandidáty potvrzené oběma metodami
        best_candidate = None
        best_score = 0

        for freq, confidence, methods in final_candidates:
            score = confidence
            if len(set(methods)) > 1:  # Potvrzeno více metodami
                score *= 1.5

            # Penalizace pro velmi vysoké frekvence (pravděpodobně harmonické)
            if freq > 1500:
                score *= 0.3
            elif freq > 800:
                score *= 0.7

            if score > best_score:
                best_score = score
                best_candidate = freq

        return best_candidate

    def _detect_harmonics(self, audio: np.ndarray, sr: int, fundamental: Optional[float]) -> List[float]:
        """Detekce harmonických složek"""
        if fundamental is None:
            return []

        harmonics = []

        # Hledej harmonické až do 8. harmonické
        for harmonic_num in range(2, 9):
            harmonic_freq = fundamental * harmonic_num

            if harmonic_freq > sr / 2:  # Nyquist limit
                break

            # Jednoduché ověření přítomnosti harmonické ve spektru
            if self._verify_harmonic_presence(audio, sr, harmonic_freq):
                harmonics.append(harmonic_freq)

        return harmonics

    def _verify_harmonic_presence(self, audio: np.ndarray, sr: int, target_freq: float) -> bool:
        """Ověří přítomnost konkrétní frekvence ve spektru"""
        # Jednoduchá spektrální analýza
        n_fft = min(4096, len(audio))
        spectrum = np.abs(np.fft.fft(audio[:n_fft]))
        freqs = np.fft.fftfreq(n_fft, 1/sr)[:n_fft//2]
        spectrum = spectrum[:n_fft//2]

        # Najdi nearest frequency bin
        freq_idx = np.argmin(np.abs(freqs - target_freq))

        # Tolerance window
        window_size = max(1, int(0.02 * len(freqs)))  # 2% tolerance
        start_idx = max(0, freq_idx - window_size)
        end_idx = min(len(spectrum), freq_idx + window_size + 1)

        # Peak v tolerance window
        local_max = np.max(spectrum[start_idx:end_idx])
        overall_mean = np.mean(spectrum)

        # Harmonic present if local peak is significantly above average
        return local_max > 3 * overall_mean

    def _calculate_confidence(self, fundamental: Optional[float], harmonics: List[float],
                            audio: np.ndarray, sr: int) -> float:
        """Výpočet confidence score pro detekovanou fundamentální frekvenci"""
        if fundamental is None:
            return 0.0

        confidence = 0.5  # Base confidence

        # Bonus za harmonické
        if len(harmonics) >= 2:
            confidence += 0.3
        elif len(harmonics) >= 1:
            confidence += 0.2

        # Bonus za frekvenci v typickém rozsahu hudebních nástrojů
        if 80 <= fundamental <= 1000:
            confidence += 0.2
        elif 1000 < fundamental <= 2000:
            confidence += 0.1

        return min(1.0, confidence)


class VelocityAnalyzer:
    """Analyzer pro velocity metriky s multiple peak detection"""

    def __init__(self, attack_duration: float = 0.5):
        self.attack_duration = attack_duration

    def analyze(self, waveform: np.ndarray, sr: int) -> VelocityAnalysisResult:
        """Kompletní velocity analýza"""

        # Flatten pro multi-channel
        if len(waveform.shape) > 1:
            audio = waveform.flatten()
        else:
            audio = waveform

        # RMS metrics
        rms_db = self._calculate_rms_db(audio)

        # Peak metrics
        peak_db = self._calculate_peak_db(audio)

        # Attack phase metrics
        attack_samples = int(sr * self.attack_duration)
        attack_samples = min(attack_samples, len(audio))

        if attack_samples > 0:
            attack_section = audio[:attack_samples]
            attack_peak_db = self._calculate_peak_db(attack_section)
            attack_rms_db = self._calculate_rms_db(attack_section)
        else:
            attack_peak_db = peak_db
            attack_rms_db = rms_db

        # Dynamic range
        dynamic_range = peak_db - rms_db if rms_db != -np.inf else 0

        return VelocityAnalysisResult(
            rms_db=rms_db,
            peak_db=peak_db,
            attack_peak_db=attack_peak_db,
            attack_rms_db=attack_rms_db,
            dynamic_range=dynamic_range
        )

    def _calculate_rms_db(self, audio: np.ndarray) -> float:
        """RMS výpočet v dB"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return -np.inf
        return 20 * np.log10(rms)

    def _calculate_peak_db(self, audio: np.ndarray) -> float:
        """Peak výpočet v dB"""
        peak = np.max(np.abs(audio))
        if peak == 0:
            return -np.inf
        return 20 * np.log10(peak)


class GlobalVelocityMapper:
    """Globální velocity mapping s multiple metrics"""

    @staticmethod
    def create_mapping(samples: List[AudioSampleData],
                      primary_metric: str = "attack_peak_db",
                      num_velocities: int = 8) -> Dict:
        """Vytvoří globální velocity mapping"""

        # Extrakce metriky ze všech vzorků
        values = []
        for sample in samples:
            if sample.velocity_analysis is None:
                continue

            if primary_metric == "attack_peak_db":
                values.append(sample.velocity_analysis.attack_peak_db)
            elif primary_metric == "peak_db":
                values.append(sample.velocity_analysis.peak_db)
            elif primary_metric == "rms_db":
                values.append(sample.velocity_analysis.rms_db)
            elif primary_metric == "dynamic_range":
                values.append(sample.velocity_analysis.dynamic_range)

        # Vyfiltruj nekonečné hodnoty
        values = [v for v in values if v != -np.inf]

        if len(values) < 2:
            logger.warning("Nedostatek dat pro velocity mapping")
            return {"thresholds": [], "min_value": 0, "max_value": 0}

        values.sort()
        min_value = min(values)
        max_value = max(values)

        # Vytvoř thresholdy
        thresholds = []
        value_range = max_value - min_value

        for i in range(num_velocities):
            threshold = min_value + (i * value_range / num_velocities)
            thresholds.append(threshold)

        return {
            "thresholds": thresholds,
            "min_value": min_value,
            "max_value": max_value,
            "metric": primary_metric
        }

    @staticmethod
    def assign_velocity(value: float, mapping: Dict) -> int:
        """Přiřadí velocity podle mapping"""
        if not mapping["thresholds"]:
            return 0

        velocity = 0
        for i, threshold in enumerate(mapping["thresholds"]):
            if value >= threshold:
                velocity = i

        return min(velocity, 7)


class SimplePitchShifter:
    """Jednoduchý pitch shifter pomocí resampling"""

    @staticmethod
    def shift(audio: np.ndarray, sr: int, semitones: float) -> Tuple[np.ndarray, int]:
        """Pitch shift změnou sample rate"""
        if abs(semitones) < 0.01:
            return audio, sr

        factor = 2 ** (semitones / 12)
        new_sr = int(sr * factor)

        try:
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                # Multi-channel
                shifted_channels = []
                for ch in range(audio.shape[1]):
                    shifted = resampy.resample(audio[:, ch], sr, new_sr)
                    shifted_channels.append(shifted)
                shifted_audio = np.column_stack(shifted_channels)
            else:
                # Mono
                audio_1d = audio.flatten() if len(audio.shape) > 1 else audio
                shifted_audio = resampy.resample(audio_1d, sr, new_sr)

                if len(audio.shape) > 1:
                    shifted_audio = shifted_audio[:, np.newaxis]

            return shifted_audio, new_sr

        except Exception as e:
            logger.error(f"Pitch shift error: {e}")
            return audio, sr


class RefactoredPitchCorrector:
    """
    Hlavní třída refaktorovaného pitch correctoru
    """

    def __init__(self, input_dir: str, output_dir: str,
                 attack_duration: float = 0.5, max_sample_duration: float = 12.0,
                 verbose: bool = False):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.attack_duration = attack_duration
        self.max_sample_duration = max_sample_duration
        self.verbose = verbose

        # Validace
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Inicializace komponent
        self.pitch_detector = HybridPitchDetector()
        self.velocity_analyzer = VelocityAnalyzer(attack_duration)
        self.pitch_shifter = SimplePitchShifter()

        # Konfigurace loggingu
        if verbose:
            logger.setLevel(logging.DEBUG)

    def process_all(self) -> None:
        """Hlavní processing pipeline"""
        try:
            logger.info("=== REFAKTOROVANÝ PITCH CORRECTOR ===")
            logger.info(f"Input: {self.input_dir}")
            logger.info(f"Output: {self.output_dir}")
            logger.info(f"Max sample duration: {self.max_sample_duration}s")
            logger.info("No pitch correction limits - always converge to nearest note")

            # Fáze 1: Načtení a analýza
            samples = self._load_and_analyze()
            if not samples:
                logger.error("No samples to process")
                return

            # Fáze 2: Velocity mapping
            velocity_mapping = self._create_velocity_mapping(samples)

            # Fáze 3: Pitch korekce a export
            self._process_and_export(samples, velocity_mapping)

            logger.info("=== PROCESSING COMPLETED ===")

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

    def _load_and_analyze(self) -> List[AudioSampleData]:
        """Načtení a analýza souborů"""
        logger.info("Phase 1: Loading and analyzing files")

        wav_files = list(self.input_dir.glob("*.wav")) + list(self.input_dir.glob("*.WAV"))

        if not wav_files:
            logger.error("No WAV files found")
            return []

        logger.info(f"Found {len(wav_files)} WAV files")

        samples = []

        for i, filepath in enumerate(wav_files, 1):
            logger.info(f"[{i}/{len(wav_files)}] Processing: {filepath.name}")

            try:
                # Načtení souboru
                waveform, sr = sf.read(str(filepath))

                # Ensure 2D format
                if len(waveform.shape) == 1:
                    waveform = waveform[:, np.newaxis]

                duration = len(waveform) / sr

                # Validace
                if sr not in [44100, 48000]:
                    logger.warning(f"Unsupported sample rate {sr} Hz, skipping")
                    continue

                if duration < 0.1:
                    logger.warning(f"File too short ({duration:.2f}s), skipping")
                    continue

                # Pitch analýza
                mono_audio = AudioUtils.to_mono(waveform)
                pitch_analysis = self.pitch_detector.detect(mono_audio, sr)

                if pitch_analysis.fundamental_freq is None:
                    logger.warning(f"Pitch detection failed for {filepath.name}")
                    continue

                if pitch_analysis.confidence < 0.3:
                    logger.warning(f"Low confidence pitch detection ({pitch_analysis.confidence:.2f})")
                    continue

                # Velocity analýza
                velocity_analysis = self.velocity_analyzer.analyze(waveform, sr)

                # Vytvoření sample data
                sample = AudioSampleData(
                    filepath=filepath,
                    waveform=waveform,
                    sample_rate=sr,
                    duration=duration,
                    pitch_analysis=pitch_analysis,
                    velocity_analysis=velocity_analysis
                )

                samples.append(sample)

                # Log analýzy
                freq = pitch_analysis.fundamental_freq
                midi = pitch_analysis.midi_note
                note_name = AudioUtils.midi_to_note_name(midi) if midi else "N/A"

                logger.info(f"  Pitch: {freq:.2f} Hz -> MIDI {midi} ({note_name})")
                logger.info(f"  Confidence: {pitch_analysis.confidence:.3f}")
                logger.info(f"  Attack Peak: {velocity_analysis.attack_peak_db:.2f} dB")
                logger.info(f"  Harmonics: {len(pitch_analysis.harmonics)}")

            except Exception as e:
                logger.error(f"Error processing {filepath.name}: {e}")
                continue

        logger.info(f"Successfully analyzed {len(samples)} samples")
        return samples

    def _create_velocity_mapping(self, samples: List[AudioSampleData]) -> Dict:
        """Vytvoření globální velocity mapy"""
        logger.info("Phase 2: Creating global velocity mapping")

        mapping = GlobalVelocityMapper.create_mapping(samples, "attack_peak_db", 8)

        if not mapping["thresholds"]:
            logger.error("Failed to create velocity mapping")
            return mapping

        # Přiřazení velocities
        for sample in samples:
            if sample.velocity_analysis:
                sample.assigned_velocity = GlobalVelocityMapper.assign_velocity(
                    sample.velocity_analysis.attack_peak_db, mapping
                )

        # Statistiky
        logger.info(f"Velocity range: {mapping['min_value']:.2f} to {mapping['max_value']:.2f} dB")

        # Distribuce velocity
        velocity_counts = defaultdict(int)
        for sample in samples:
            if sample.assigned_velocity is not None:
                velocity_counts[sample.assigned_velocity] += 1

        logger.info("Velocity distribution:")
        for vel in range(8):
            count = velocity_counts[vel]
            logger.info(f"  Velocity {vel}: {count} samples")

        return mapping

    def _process_and_export(self, samples: List[AudioSampleData], velocity_mapping: Dict) -> None:
        """Zpracování a export souborů s round-robin trackingem"""
        logger.info("Phase 3: Processing and exporting")

        total_exported = 0

        # Tracking kolik vzorků už bylo vytvořeno pro každou MIDI+velocity kombinaci
        sample_counters = defaultdict(int)  # key: (midi, velocity), value: počet vzorků

        for i, sample in enumerate(samples, 1):
            if sample.assigned_velocity is None or sample.pitch_analysis.fundamental_freq is None:
                continue

            logger.info(f"[{i}/{len(samples)}] Exporting: {sample.filepath.name}")

            # Určení cílové MIDI noty
            detected_midi = sample.pitch_analysis.midi_note
            target_midi = self._find_target_midi_note(sample.pitch_analysis.fundamental_freq)

            if target_midi is None:
                logger.warning("Cannot determine target MIDI note")
                continue

            # Výpočet korekce
            target_freq = AudioUtils.midi_to_freq(target_midi)
            detected_freq = sample.pitch_analysis.fundamental_freq
            semitone_correction = 12 * np.log2(target_freq / detected_freq)

            # DŮLEŽITÉ: Žádné limity korekce!
            # Pokud je korekce velká, problém je v pitch detekci, ne ve vzorku

            # Pouze informativní warning pro velké korekce (indikátor špatné detekce)
            if abs(semitone_correction) > 1.0:
                logger.warning(f"Large correction {semitone_correction:.3f} semitones - possible pitch detection error")
                logger.warning(f"Detected: {detected_freq:.2f} Hz -> Target: {target_freq:.2f} Hz")
                # Ale POKRAČUJEME ve zpracování!

            sample.target_midi_note = target_midi
            sample.pitch_correction_semitones = semitone_correction

            # Určení round-robin indexu
            rr_key = (target_midi, sample.assigned_velocity)
            rr_index = sample_counters[rr_key]
            sample_counters[rr_key] += 1

            note_name = AudioUtils.midi_to_note_name(target_midi)
            if rr_index == 0:
                logger.info(f"  MIDI {target_midi} ({note_name}) -> velocity {sample.assigned_velocity}")
            else:
                logger.info(f"  MIDI {target_midi} ({note_name}) -> velocity {sample.assigned_velocity} [round-robin {rr_index+1}]")
            logger.info(f"  Pitch correction: {semitone_correction:+.3f} semitones")

            # Aplikace pitch korekce
            if abs(semitone_correction) > 0.01:
                corrected_audio, corrected_sr = self.pitch_shifter.shift(
                    sample.waveform, sample.sample_rate, semitone_correction
                )
            else:
                corrected_audio = sample.waveform
                corrected_sr = sample.sample_rate

            # Export pro oba target sample rates
            target_rates = [(44100, 'f44'), (48000, 'f48')]

            for target_sr, sr_suffix in target_rates:
                # Konverze sample rate
                if corrected_sr != target_sr:
                    final_audio = self._convert_sample_rate(corrected_audio, corrected_sr, target_sr)
                else:
                    final_audio = corrected_audio

                # Crop na maximální délku s ohledem na skutečný sample rate
                final_audio = self._crop_audio(final_audio, target_sr, self.max_sample_duration)

                # Informace o cropu (porovnání před a po cropu)
                final_duration = len(final_audio) / target_sr
                original_duration = len(corrected_audio) / corrected_sr
                if final_duration < original_duration:
                    logger.info(f"  Cropped: {original_duration:.2f}s -> {final_duration:.2f}s")

                # Generování názvu souboru s round-robin indexem
                output_path = self._generate_filename(target_midi, sample.assigned_velocity, sr_suffix, rr_index)

                # Uložení
                try:
                    sf.write(str(output_path), final_audio, target_sr)
                    logger.info(f"  Saved: {output_path.name}")
                    total_exported += 1
                except Exception as e:
                    logger.error(f"Export error: {e}")

        # Statistiky round-robin vzorků
        logger.info("\nRound-robin statistics:")
        for (midi, velocity), count in sorted(sample_counters.items()):
            if count > 1:
                note_name = AudioUtils.midi_to_note_name(midi)
                logger.info(f"  MIDI {midi} ({note_name}) velocity {velocity}: {count} samples")

        logger.info(f"Total exported: {total_exported} files")

    def _find_target_midi_note(self, freq: float) -> Optional[int]:
        """Najde nejbližší MIDI notu pro danou frekvenci"""
        try:
            midi_float = 12 * np.log2(freq / 440.0) + 69
            # Vždy zaokrouhli na nejbližší MIDI notu - žádné limity!
            midi_note = int(np.round(midi_float))

            # Pouze validace rozsahu MIDI (0-127)
            if 0 <= midi_note <= 127:
                return midi_note
            else:
                logger.warning(f"MIDI note {midi_note} outside valid range for frequency {freq:.2f} Hz")
                return None

        except (ValueError, OverflowError):
            return None

    def _convert_sample_rate(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Konverze sample rate"""
        try:
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                # Multi-channel
                converted_channels = []
                for ch in range(audio.shape[1]):
                    converted = resampy.resample(audio[:, ch], from_sr, to_sr)
                    converted_channels.append(converted)
                return np.column_stack(converted_channels)
            else:
                # Mono
                audio_1d = audio.flatten() if len(audio.shape) > 1 else audio
                converted = resampy.resample(audio_1d, from_sr, to_sr)
                return converted[:, np.newaxis] if len(audio.shape) > 1 else converted
        except Exception as e:
            logger.error(f"Sample rate conversion error: {e}")
            return audio

    def _generate_filename(self, midi: int, velocity: int, sr_suffix: str,
                          sample_index: int = 0) -> Path:
        """Generuje strukturovaný název souboru pro round-robin samples"""

        # Formát: m060-vel3-f44-rr2.wav (round-robin po frekvenci)
        if sample_index == 0:
            # První vzorek bez round-robin označení
            base_name = f"m{midi:03d}-vel{velocity}-{sr_suffix}"
        else:
            # Další vzorky s round-robin indexem za frekvencí
            base_name = f"m{midi:03d}-vel{velocity}-{sr_suffix}-rr{sample_index+1}"

        output_path = self.output_dir / f"{base_name}.wav"
        return output_path

    def _crop_audio(self, audio: np.ndarray, actual_sr: int, max_duration: float = 12.0) -> np.ndarray:
        """
        Crop audio na maximální délku na základě skutečného sample rate.

        Args:
            audio: Audio data (mono nebo stereo)
            actual_sr: Skutečný sample rate audio dat (po pitch shift)
            max_duration: Maximální délka v sekundách
        """
        # Výpočet maximálního počtu vzorků na základě SKUTEČNÉHO sample rate
        max_samples = int(actual_sr * max_duration)

        if len(audio) <= max_samples:
            return audio

        # Crop ze začátku - zachová attack fázi
        if len(audio.shape) > 1:
            # Multi-channel
            return audio[:max_samples, :]
        else:
            # Mono
            return audio[:max_samples]


def parse_arguments():
    """Parsování argumentů příkazové řádky"""
    parser = argparse.ArgumentParser(
        description="""
Refaktorovaný Pitch Corrector s pokročilou harmonickou detekcí

Klíčové vylepšení:
- Hybridní pitch detection (spektrální + YIN)
- Robustní harmonická filtrace
- Globální velocity mapping založený na attack peak detection
- Modulární architektura s error handling
- Comprehensive logging

Příklad použití:
python refactored_pitch_corrector.py --input-dir ./samples --output-dir ./output --verbose
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--input-dir', required=True,
                       help='Cesta k vstupnímu adresáři s WAV soubory')
    parser.add_argument('--output-dir', required=True,
                       help='Cesta k výstupnímu adresáři')
    parser.add_argument('--attack-duration', type=float, default=0.5,
                       help='Délka attack fáze pro velocity detection (default: 0.5s)')
    parser.add_argument('--max-duration', type=float, default=12.0,
                       help='Maximální délka výstupních vzorků v sekundách (default: 12.0s)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging pro debugging')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    try:
        corrector = RefactoredPitchCorrector(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            attack_duration=args.attack_duration,
            max_sample_duration=args.max_duration,
            verbose=args.verbose
        )

        corrector.process_all()

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)