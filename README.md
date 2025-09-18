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

## Vytvoření velocity mapy

Program vytváří globální velocity mapu na základě analýzy všech vstupních vzorků (ze všech not). 
To probíhá ve fázi 2 zpracování (metoda create_velocity_mappings třídy PitchCorrectorWithVelocityMapping):

- Sběr dat: Z každého vzorku se extrahuje hodnota attack peak dB (vrcholová hlasitost v prvních 0.5 sekundách, konfigurovatelná parametrem --attack-duration). Tyto hodnoty se shromáždí do seznamu ze všech platných vzorků.
- Výpočet rozsahu: Seřadí se všechny peak hodnoty, určí se minimální a maximální hodnota (min_peak, max_peak). Rozsah se rozdělí rovnoměrně na 8 skupin (velocity 0–7).
- Vytvoření hranic (thresholds): Každá skupina má dolní hranici (např. vel0: min_peak, vel1: min_peak + krok, atd.). Krok = (max_peak - min_peak) / 8. Tím vznikne jednotná škála pro celou sadu vzorků, která se adaptuje na dynamický rozsah nahrávek.
- Statistiky: Program vypočítá distribuce (počet vzorků na velocity, min/max/avg peak) globálně i pro jednotlivé MIDI noty. Zobrazí se noty s více velocity úrovněmi.

Nejslabší vzorky dostanou vel0, nejsilnější vel7 - bez ohledu na notu.

## Práce s velocity mapou

Po vytvoření se mapa používá v následujících krocích:

- Přiřazení velocity vzorkům: Každému vzorku se přiřadí velocity (0–7) podle jeho attack peak dB vzhledem k thresholds (metoda assign_velocity_from_global_map). Např. pokud peak překročí hranici vel3, dostane vel3.
- Export souborů: Ve fázi 3 (process_and_export) se velocity používá pro pojmenování výstupních WAV souborů (formát: m<MIDI>-vel<velocity>-f44.wav nebo f48 pro 44.1/48 kHz). Každý vzorek se uloží dvakrát (pro obě vzorkovací frekvence), s pitch korekcí, ale velocity ovlivňuje jen název a seskupení.
- Finální shrnutí: Program zobrazí statistiky mapy (rozsah dB, distribuce) v závěrečném výstupu.

Pokud je celkový rozsah peak hodnot ze všech vzorků menší než 1 dB, program přiřadí všem vzorkům stejnou velocity (typicky vel0), protože dynamický rozsah je považován za nedostatečný pro rozdělení do více skupin. V metodě ```create_global_velocity_mapping``` třídy ```VelocityMapper```, konkrétně v podmínce ```if peak_range < 1.0```.

## Tuning a export

Tato fáze (metoda ```process_and_export```) zpracovává analyzované vzorky s přiřazenou velocity a vytváří výstupní WAV soubory. 
Probíhá po vytvoření globální velocity mapy.

Hlavní kroky:

- Filtrace vzorků: Vyberou se pouze validní vzorky (ty s přiřazenou velocity). Pokud jich není, proces končí chybou.

Pro každý vzorek:

- Vypočítá se rozdíl v půltónech (semitone_shift) mezi detekovanou a cílovou frekvencí MIDI noty.
- Kontroluje se, zda korekce nepřekračuje limit (±1 půltón); pokud ano, vzorek se přeskočí.
- Aplikuje se jednoduchý pitch shift (pomocí SimplePitchShifter), který upraví výšku tónu, ale změní délku vzorku.
- Pro dvě cílové vzorkovací frekvence (44.1 kHz a 48 kHz):
- Pokud je potřeba, provede se resample na cílovou frekvenci.
- Vygeneruje se unikátní název souboru (např. m060-vel3-f44.wav), s případným přídomkem ```-next1``` pro duplicity.
- Uloží se WAV soubor (pomocí ```soundfile.write```).
- Výstup: Zobrazí se progress pro každý soubor (MIDI, velocity, korekce, délka). Na konci vrátí celkový počet vytvořených souborů pro finální shrnutí.

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
- Program zahazuje vzorky s nedetekovatelným pitchem.
- Globální velocity mapping zajišťuje konzistenci napříč všemi notami na základě celkového rozsahu nahrávky.
- V verbose režimu je výstup detailnější.
