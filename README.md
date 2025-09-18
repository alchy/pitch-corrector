# Pitch Corrector s Globálním Velocity Mappingem

## Popis
Tento program slouží k automatické korekci výšky tónu (pitch) a mapování dynamiky (velocity) u vzorků hudebních nástrojů. 
Používá pokročilou YIN detekci pro určení výšky tónu, globální velocity mapping na základě peak detekce v attack fázi 
a adaptivní oktávovou korekci pro extrémní frekvence. Program je optimalizován pro WAV soubory s vzorkovacími 
frekvencemi 44.1 kHz nebo 48 kHz a exportuje upravené vzorky do dvou formátů.

Klíčové vlastnosti:

- Detekce výšky tónu s oktávovou korekcí pro nízké/vysoké tóny.
- Globální velocity mapping napříč všemi vzorky (0-7 úrovní) na základě attack peak detekce.
- Pitch shifting, který mění délku vzorku pro realistický efekt.
- Export do 44.1 kHz a 48 kHz s unikátními názvy souborů (např. `m060-vel3-f44.wav`).
- Podpora pro mono i stereo soubory.
- Omezení maximální korekce na ±1 půltón.

Program probíhá ve třech fázích: načtení a analýza, tvorba velocity mapy, tuning a export.

## Požadavky a Instalace

- Python 3.12 nebo vyšší.
- Závislosti: `soundfile`, `numpy`, `resampy`, `collections`, `statistics`, `pathlib`, `tqdm`, `logging`.

Instalace závislostí:

```
pip install soundfile numpy resampy tqdm
```

Program nevyžaduje další instalaci – stačí spustit Python skript.

## Použití

Spusťte program z příkazového řádku s povinnými argumenty pro vstupní a výstupní adresáře.

### Argumenty příkazového řádku

- `--input-dir`: Cesta k adresáři se vstupními WAV soubory (povinné).
- `--output-dir`: Cesta k výstupnímu adresáři (povinné).
- `--attack-duration`: Délka attack fáze pro peak detekci v sekundách (výchozí: 0.5).
- `--verbose`: Zapne podrobný výstup pro ladění (vypne progress bary).

### Příklad spuštění

```
python pitch_corrector.py --input-dir ./samples --output-dir ./output --attack-duration 0.5 --verbose
```

Toto zpracuje všechny WAV soubory v `./samples`, vytvoří globální velocity mapu a uloží upravené vzorky do `./output`.

## Výstup

- Výstupní soubory mají formát `m<MIDI>-vel<VELOCITY>-f<SR>.wav` (např. `m060-vel3-f44.wav` pro MIDI 60, velocity 3, 44.1 kHz).
- Pokud existuje soubor se stejným názvem, přidá se suffix `-next1`, `-next2` atd.
- Program vypisuje průběh, statistiky velocity mapy a finální shrnutí.

## Poznámky
- Program zahazuje vzorky s nedetekovatelným pitchem, příliš velkou korekcí nebo mimo MIDI rozsah 0-127.
- Globální velocity mapping zajišťuje konzistenci napříč všemi notami na základě celkového rozsahu nahrávky.
- V verbose režimu je výstup detailnější pro lepší ladění, ale bez progress barů.
