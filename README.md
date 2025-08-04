# Pitch Corrector

Program pro automatickou korekci ladění (pitch) WAV souborů na standardní hudební ladění (A4 = 440 Hz). Nástroj detekuje fundamentální frekvenci vstupních audio souborů, identifikuje nejbližší MIDI notu a upravuje ladění tak, aby odpovídalo přesně této notě.

## Funkce

- **Automatická detekce pitch**: Pokročilá detekce fundamentální frekvence s více metodami pro maximální přesnost
- **Podpora stereo i mono**: Zpracovává jak stereo, tak mono WAV soubory
- **Rozsah klavíru**: Optimalizováno pro detekci not v rozsahu klavíru (A0 - C8, 27.5 Hz - 4186 Hz)
- **Zachování kvality**: Zachovává původní hlasitost a kvalitu zvuku
- **Inteligentní naming**: Výstupní soubory jsou pojmenovány podle MIDI čísla, noty a hlasitosti
- **Paměťová optimalizace**: Volitelná podpora librosa pro efektivnější práci s pamětí

## Požadavky

### Základní požadavky
```
torch
torchaudio
numpy
```

### Doporučené (pro lepší výkon)
```
librosa
resampy
```

## Instalace

1. Naklonujte repository:
```bash
git clone <repository-url>
cd pitch-corrector
```

2. Nainstalujte základní závislosti:
```bash
pip install torch torchaudio numpy
```

3. (Doporučeno) Nainstalujte librosa pro lepší paměťovou efektivitu:
```bash
pip install librosa resampy
```

## Použití

### Základní použití
```bash
python pitch_corrector.py --input-dir ./vstupni_soubory --output-dir ./vystupni_soubory
```

### Parametry

- `--input-dir` (povinný): Cesta k adresáři obsahujícímu vstupní WAV soubory
- `--output-dir` (povinný): Cesta k adresáři pro uložení opravených souborů

### Příklady

1. **Zpracování adresáře s audio soubory:**
```bash
python pitch-corrector.py --input-dir ./samples_in --output-dir ./samples_out_tuned
```

2. **Zobrazení nápovědy:**
```bash
python pitch_corrector.py --help
```

## Formát vstupních souborů

Program akceptuje WAV soubory s následujícími parametry:
- **Formát**: WAV (mono nebo stereo)
- **Vzorkovací frekvence**: 44.1 kHz
- **Maximální délka**: 12 sekund
- **Frekvence**: Rozsah klavíru (A0 - C8, ~27.5 Hz - 4186 Hz)

## Formát výstupních souborů

Opravené soubory jsou uloženy s názvem ve formátu:
```
mXXX-NOTA-DbLvl±XX.wav
```

Kde:
- `XXX` = MIDI číslo noty (např. `060` pro C4)
- `NOTA` = Název noty s oktávou (např. `C4`, `F#3`)
- `±XX` = Hlasitost v dB (např. `-023`, `+005`)

### Příklady názvů souborů:
- `m060-C4-DbLvl-023.wav` - střední C, hlasitost -23 dB
- `m069-A4-DbLvl-018.wav` - A4 (440 Hz), hlasitost -18 dB
- `m072-C5-DbLvl+002.wav` - C5, hlasitost +2 dB

## Algoritmus detekce pitch

Program používá tři metody detekce pro maximální přesnost:

1. **Torchaudio detector**: Vestavěný detektor PyTorch/torchaudio
2. **Librosa piptrack**: Pokročilá detekce s analýzou harmonických (pokud je k dispozici)
3. **Autokorelace**: Záložní metoda založená na autokorelační analýze

### Preprocessing
- Konverze na mono (průměrování kanálů u stereo)
- Normalizace na konzistentní úroveň (-20 dB)
- High-pass filtr pro odstranění nízkofrekvenčního šumu

## Výstup programu

Program zobrazuje podrobné informace o zpracování:

```
Používám zařízení: cpu
Zpracovávám sample001.wav
Formát: stereo
Detekovaná frekvence: 261.63 Hz
MIDI: 60.00 -> 60, posun: 0.000 půltónů
Žádný pitch shift není potřeba
Zpracováno: sample001.wav -> m060-C4-DbLvl-023.wav, pitch: 261.63 Hz -> 261.63 Hz, stereo
--------------------------------------------------
```

## Řešení problémů

### Chybové hlášky

**"Librosa nebo resampy není k dispozici"**
- Program bude pokračovat s torchaudio, ale doporučuje se instalace librosa pro lepší výkon
- Řešení: `pip install librosa resampy`

**"Soubor má neplatnou vzorkovací frekvenci"**
- Soubor nemá 44.1 kHz vzorkovací frekvenci
- Řešení: Převeďte soubor na 44.1 kHz pomocí audio editoru

**"Soubor je delší než 12 sekund"**
- Program nepodporuje dlouhé soubory z důvodu paměťových limitů
- Řešení: Rozdělte soubor na kratší segmenty

**"Nelze detekovat platnou frekvenci"**
- Audio neobsahuje jasně definovanou fundamentální frekvenci nebo je mimo rozsah klavíru
- Řešení: Ověřte kvalitu nahrávky a přítomnost tónu
