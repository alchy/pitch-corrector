## Popis programu
Tento program slouží k automatické korekci pitch (výšky tónu) a mapování velocity (hlasitosti) pro vzorky hudebních nástrojů ve formátu WAV. Používá pokročilou verzi algoritmu YIN pro detekci pitch s adaptivní oktávovou korekcí a analýzou sustain fáze pro lepší přesnost. Program seskupuje vzorky podle MIDI not a RMS hlasitosti, aplikuje jednoduchý pitch shift (který mění délku vzorku) a exportuje výsledky v dvou vzorkovacích frekvencích (44.1 kHz a 48 kHz).

Klíčové funkce:
- Detekce pitch s oktávovou korekcí pro nízké/vysoké tóny.
- Mapování velocity (0-7) na základě seskupování RMS hodnot (max 8 vrstev).
- Korekce na nejbližší půltón.
- Podpora mono i stereo souborů.
- Verbose režim pro detailní výstup.

Program je optimalizován pro vzorky hudebních nástrojů s dynamickým envelope, ale lze ho použít i pro jiné audio vzorky.

Autor: Refaktorovaná verze pro obecné hudební nástroje  
Datum: 2025  

## Instalace a závislosti
Program je napsán v Pythonu 3 a vyžaduje následující knihovny. Nainstalujte je pomocí pip:

```
pip install soundfile numpy resampy tqdm
```

Další standardní moduly (jako collections, statistics, pathlib, logging) jsou součástí Pythonu.

## Použití
Spusťte program z příkazové řádky s povinnými argumenty pro vstupní a výstupní adresář.

### Argumenty příkazové řádky
- `--input-dir`: Cesta k adresáři s vstupními WAV soubory (povinné).
- `--output-dir`: Cesta k výstupnímu adresáři (povinné).
- `--grouping-threshold`: Práh pro seskupování RMS hodnot v dB (výchozí: 1.5).
- `--verbose`: Zapne podrobný výstup (bez progress barů pro lepší čitelnost).

### Příklad spuštění
```
python pitch_corrector.py --input-dir ./samples --output-dir ./output --grouping-threshold 2.0 --verbose
```

Program prochází třemi fázemi:
1. Načtení a analýza souborů (detekce pitch, RMS).
2. Tvorba velocity map pro každou MIDI notu.
3. Tuning pitch a export souborů (s unikátními názvy jako `m060-vel3-f44.wav`).

Výstupní soubory mají formát: `m<MIDI>-vel<velocity>-f<sr>[-nextX].wav`, kde:
- MIDI je číslo noty (např. 060 pro C4).
- Velocity je 0-7 (od tichého po hlasité).
- sr je 44 nebo 48 (pro 44.1 kHz nebo 48 kHz).
- Pokud existuje duplicitní název, přidá se `-nextX`.

## Poznámky a omezení
- Podporované vzorkovací frekvence: Pouze 44.1 kHz nebo 48 kHz.
- Minimální délka vzorku: 0.1 s.
- Pokud nelze detekovat pitch, vzorek se přeskočí.
- Velocity mapping vyřadí extrémní vzorky, pokud by překročilo 8 skupin.
- Program mění délku vzorku při pitch shiftu, což je realistické pro mnoho nástrojů (kratší vysoké tóny, delší nízké). 
- .
