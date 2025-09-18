# README.md pro Program Korekce Pitch a Velocity Mapping

## Popis
Tento Python program slouží k automatické korekci pitch (tónové výšky) a mapování velocity (hlasitosti) pro vzorky hudebních nástrojů. Používá pokročilou verzi YIN algoritmu pro detekci pitch s adaptivní oktávovou korekcí a analýzou sustain fáze. Program zpracovává WAV soubory, odstraňuje outliery, vytváří velocity vrstvy (0-7) na základě RMS hlasitosti a exportuje korigované vzorky v 44.1 kHz a 48 kHz formátech.

Klíčové funkce:
- Detekce pitch s oktávovou korekcí pro nízké/vysoké tóny.
- Velocity mapping pro každou MIDI notu.
- Jednoduchý pitch shift (mění délku vzorku pro realističnost).
- Omezení maximální korekce na ±1 půltón.
- Podpora stereo souborů a automatická normalizace.

Program je optimalizován pro vzorky hudebních nástrojů s dynamickým envelope. Vstupní soubory musí mít vzorkovací frekvenci 44.1 kHz nebo 48 kHz.

## Požadavky
- Python 3.8+
- Knihovny: `numpy`, `soundfile`, `resampy`, `scipy` (pro některé funkce, i když není explicitně importován), `argparse`, `collections`, `statistics`, `pathlib`.

Instalace knihoven:
```
pip install numpy soundfile resampy
```

## Použití
Spusťte program přes příkazový řádek s následujícími argumenty:

- `--input-dir`: Cesta k adresáři s vstupními WAV soubory (povinný).
- `--output-dir`: Cesta k výstupnímu adresáři (povinný).
- `--outlier-threshold`: Práh pro odstranění outlierů v dB (výchozí: 8.0).
- `--verbose`: Zapne podrobný výstup pro ladění pitch detekce.

Příklad:
```
python pitch_corrector.py --input-dir ./samples --output-dir ./output --verbose
```

Výstupní soubory mají formát: `m<MIDI>-vel<VELOCITY>-f44.wav` nebo `f48.wav`, s případným příponou `-nextN` pro unikátnost.

## Proces zpracování
1. **Načtení a analýza**: Program načte všechny WAV soubory z vstupního adresáře, pro každý soubor provede pokročilou detekci pitch pomocí YIN algoritmu (s adaptivní oktávovou korekcí a výběrem sustain sekce pro stabilnější výsledky), vypočítá odpovídající MIDI notu (s kontrolou maximální odchylky ±1 půltón), určí RMS hlasitost (pro pozdější velocity mapping) a uloží tyto data do interních struktur pro další fáze. Pokud pitch nelze detekovat nebo je odchylka příliš velká, soubor je přeskočen.

2. **Velocity mapping**: Vzorky jsou seskupeny podle MIDI not (každá nota zpracována samostatně), pro každou skupinu se vypočítají RMS hodnoty, odstraní se outliery na základě mediánové odchylky (s nastavitelným prahem), a poté se RMS hodnoty lineárně mapují do velocity vrstev 0-7 (od nejtišší po nejsilnější). Každému vzorku je přiřazena odpovídající velocity; outliery jsou vyřazeny z dalšího zpracování.

3. **Tuning a export**: Pro každý platný vzorek se znovu potvrdí pitch, aplikuje se jednoduchý pitch shift (pomocí resamplingu, což mění délku vzorku pro přirozený efekt), normalizuje se audio, konvertuje se do cílových vzorkovacích frekvencí (44.1 kHz a 48 kHz) a uloží se do výstupního adresáře s unikátním názvem obsahujícím MIDI, velocity a frekvenci. Proces zahrnuje chybovou kontrolu při ukládání.

## Omezení
- Podporuje pouze WAV soubory s 44.1/48 kHz.
- Žádná podpora pro instalaci dodatečných balíčků (používá vestavěné knihovny).
- Maximální korekce ±1 půltón – větší odchylky jsou zahazovány.

## Autor a verze
- Autor: Refaktorovaná verze pro obecné hudební nástroje.
- Datum: 2025.

## Poznámky k programu
Při kontrole programu jsem nenašel zásadní nedostatky – je dobře strukturovaný, s komentáři a error handlingem. Menší návrhy na zlepšení:
- Uložení detected_pitch do AudioSample by ušetřilo re-detekci v fázi 3.
- Přidání podpory pro více formátů (např. FLAC) by bylo užitečné, ale není vyžadováno.
- V verbose módu by mohlo být více logů o oktávových korekcích. Pokud chceš tyto změny implementovat, navrhnu je nejdříve.