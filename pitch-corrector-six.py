import os
import torch
import torchaudio
import numpy as np
import argparse
import resampy


class AudioUtils:
    MIDI_TO_NOTE = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
        6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'
    }

    @staticmethod
    def freq_to_midi(freq):
        return 12 * np.log2(freq / 440.0) + 69

    @staticmethod
    def midi_to_freq(midi):
        return 440.0 * 2 ** ((midi - 69) / 12)

    @staticmethod
    def midi_to_note_name(midi):
        octave = (midi // 12) - 1
        note_idx = midi % 12
        note = AudioUtils.MIDI_TO_NOTE[note_idx]
        if len(note) == 1:
            note += '_'
        return f"{note}{octave}"

    @staticmethod
    def calculate_rms_db(waveform):
        rms = torch.sqrt(torch.mean(waveform ** 2))
        return 20 * torch.log10(rms + 1e-10).item()


class PitchDetector:
    def __init__(self, sample_rate=44100, downsample_factor=1):
        self.sample_rate = sample_rate
        self.downsample_factor = downsample_factor
        self.min_duration = 0.1
        self.fmin = 32.0
        self.fmax = 4186.0
        self.threshold = 0.1
        self.frame_length = 2048
        self.hop_size = 512

    def normalize(self, waveform, target_db=-20.0):
        rms = torch.sqrt(torch.mean(waveform ** 2))
        target_rms = 10 ** (target_db / 20)
        return waveform * (target_rms / (rms + 1e-10))

    def preprocess(self, waveform):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return self.normalize(waveform)

    def yin_pitch(self, audio, effective_sr):
        def difference_function(audio, max_tau):
            diff = np.zeros(max_tau)
            for tau in range(max_tau):
                diff[tau] = np.sum((audio[:len(audio)-tau] - audio[tau:]) ** 2)
            return diff

        def cmndf(diff, max_tau):
            cmndf = np.zeros(max_tau)
            cmndf[0] = 1.0
            cumsum = np.cumsum(diff[1:])
            for tau in range(1, max_tau):
                cmndf[tau] = diff[tau] * tau / (cumsum[tau-1] + 1e-10)
            return cmndf

        def parabolic_interpolation(cmndf, tau):
            if tau == 0 or tau == len(cmndf) - 1:
                return tau
            left = cmndf[tau-1] if tau > 0 else cmndf[tau]
            center = cmndf[tau]
            right = cmndf[tau+1] if tau < len(cmndf)-1 else cmndf[tau]
            denom = 2 * (left - 2 * center + right)
            if abs(denom) < 1e-10:
                return tau
            delta = (left - right) / denom
            return tau + delta / 2

        max_tau = int(effective_sr / self.fmin)
        min_tau = int(effective_sr / self.fmax)
        diff = difference_function(audio, max_tau)
        cmndf_vals = cmndf(diff, max_tau)

        for tau in range(min_tau, max_tau):
            if cmndf_vals[tau] < self.threshold:
                tau_interp = parabolic_interpolation(cmndf_vals, tau)
                if tau_interp <= 0:
                    return None
                return effective_sr / tau_interp
        return None

    def detect(self, waveform):
        duration = waveform.shape[1] / self.sample_rate
        if duration < self.min_duration:
            print(f"[DEBUG] Signál je příliš krátký ({duration:.3f}s)")
            return None

        processed = self.preprocess(waveform)
        audio = processed.mean(dim=0).cpu().numpy()

        ds_factor = self.downsample_factor
        if ds_factor != 1:
            print(f"[DEBUG] Downsampling aktivní: faktor {ds_factor}")
            audio = resampy.resample(audio, self.sample_rate, self.sample_rate // ds_factor)
            effective_sr = self.sample_rate // ds_factor
        else:
            effective_sr = self.sample_rate

        try:
            pitches = []
            for i in range(0, len(audio) - self.frame_length, self.hop_size):
                frame = audio[i:i + self.frame_length]
                pitch = self.yin_pitch(frame, effective_sr)
                if pitch is not None and self.fmin <= pitch <= self.fmax:
                    pitches.append(pitch * ds_factor)

            if not pitches:
                print("[DEBUG] Žádné validní frekvence nebyly detekovány.")
                return None

            median_pitch = np.median(pitches)
            print(f"[DEBUG] Detekovaný pitch (Hz): {median_pitch:.2f}")
            return median_pitch
        except Exception as e:
            print(f"[DEBUG] Chyba při detekci výšky tónu: {e}")
            return None


def apply_pitch_with_resampy(waveform, sr, semitone_shift):
    factor = 2 ** (semitone_shift / 12)
    target_sr = int(sr * factor)

    output_channels = []
    for ch in waveform.numpy():
        shifted = resampy.resample(ch, sr, target_sr)
        shifted_back = resampy.resample(shifted, target_sr, sr)
        output_channels.append(shifted_back)

    return torch.tensor(np.stack(output_channels)).float()


class PitchCorrector:
    def __init__(self, input_dir, output_dir, downsample_factor=1):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.detector = PitchDetector(downsample_factor=downsample_factor)
        os.makedirs(self.output_dir, exist_ok=True)

    def correct_all(self):
        for file in os.listdir(self.input_dir):
            if file.lower().endswith('.wav'):
                self.process_file(file)

    def process_file(self, filename):
        path_in = os.path.join(self.input_dir, filename)
        print(f"[INFO] Zpracovávám {filename}")
        try:
            waveform, sr = torchaudio.load(path_in)
        except Exception as e:
            print(f"[ERROR] Chyba načítání {filename}: {e}")
            return

        if sr != 44100:
            print(f"[WARNING] Přeskočeno {filename}: očekáván sample rate 44100 Hz, nalezen {sr}")
            return

        if waveform.shape[1] / sr > 12:
            print(f"[WARNING] Přeskočeno {filename}: soubor delší než 12 sekund")
            return

        original = waveform.clone()
        pitch = self.detector.detect(waveform)

        if pitch is None:
            print(f"[WARNING] Nepodařilo se detekovat výšku tónu u {filename}")
            return

        midi = round(AudioUtils.freq_to_midi(pitch))
        target_freq = AudioUtils.midi_to_freq(midi)
        semitone_shift = 12 * np.log2(target_freq / pitch)

        note = AudioUtils.midi_to_note_name(midi)
        db_level = round(AudioUtils.calculate_rms_db(original))

        print(f"[DEBUG] Původní pitch: {pitch:.2f} Hz -> Cíl: {target_freq:.2f} Hz (MIDI {midi}, nota {note})")
        print(f"[DEBUG] Posun o {semitone_shift:.2f} půltónů | RMS: {db_level} dB")

        if abs(semitone_shift) < 0.01:
            corrected = original
            print(f"[INFO] {filename}: není potřeba měnit pitch")
        else:
            try:
                corrected = apply_pitch_with_resampy(original, sr, semitone_shift)
            except Exception as e:
                print(f"[ERROR] Chyba při pitch shift u {filename}: {e}")
                return

        out_name = f"m{midi:03d}-{note}-DbLvl{db_level:+03d}.wav"
        out_path = os.path.join(self.output_dir, out_name)

        try:
            torchaudio.save(out_path, corrected, sr)
            print(f"[SUCCESS] Hotovo: {filename} -> {out_name}")
        except Exception as e:
            print(f"[ERROR] Chyba ukládání {out_name}: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detekce a korekce výšky tónu WAV souborů na základě YIN algoritmu.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--input-dir', required=True, help='Cesta ke vstupním WAV souborům.')
    parser.add_argument('--output-dir', required=True, help='Cesta pro výstupní WAV soubory.')
    parser.add_argument('--downsample-factor', type=int, default=1, help='Faktor downsamplování pro detekci pitch (např. 2).')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    corrector = PitchCorrector(args.input_dir, args.output_dir, downsample_factor=args.downsample_factor)
    corrector.correct_all()
