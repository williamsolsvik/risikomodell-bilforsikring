# Risikomodell og tariff for bilansvar

En læringsøvelse i prising av bilansvar: skadefrekvens og skadestørrelse modelleres
med GAM, og de glatte kurvene gjøres om til en tariff med grunnpris og faktorer per
klasse. Den nye tariffen sammenlignes med en enkel GLM med grupperte variabler, som
spiller rollen som «dagens tariff».

## Data

Det franske datasettet freMTPL2 fra OpenML: freMTPL2freq (id 41214) med 678 013
poliser og freMTPL2sev (id 41215) med 26 639 skadebeløp. Lisensen er CC0. Alle
beløp er i euro. Dataene lastes ned automatisk til `data/` første gang
`00_lastinn.py` kjøres, og md5-summen kontrolleres.

## Fremgangsmåte

| Steg | Skript | Hva |
| --- | --- | --- |
| 0 | `00_lastinn.py` | Laster ned og leser dataene, lagrer som parquet |
| 1 | `01_utforsk.py` | Sjekker datakvalitet, setter av 20 % testdata, lager enveis-tabeller |
| 2 | `02_rens.py` | Definerer skader, kapper storskader ved 99,5-persentilen og beregner påslag |
| 3 | `03_baseline_glm.py` | Poisson-GLM med grupperte variabler («dagens tariff») |
| 4 | `04_frekvens_gam.py` | Poisson-GAM for skadefrekvens |
| 5 | `05_alvorlighet_gam.py` | Gamma-GAM for skadestørrelse |
| 6 | `06_evaluering.py` | Sammenligner GAM og GLM på test: deviance, A/E, double lift, Gini med bootstrap |
| 7 | `07_tariff.py` | Gjør kurvene om til tariffklasser og kalibrerer grunnprisen |
| 8 | `08_endringsanalyse.py` | Sammenligner ny og gammel premie per polise |

## Hovedfunn

Detaljer, figurer og metode står i [rapport.md](rapport.md).

- **GAM rangerer skadefrekvens litt bedre enn GLM.** Gini på test er 0,3027 mot
  0,2969, og GAM var best i 99 % av 200 bootstrap-trekk. For skadekostnad er
  forskjellen ikke sikker (GAM best i 74 % av trekkene).
- **Forbedringen er liten.** Poisson-deviance på test forbedres med 4,95 % mot
  nullmodellen, mot 4,71 % for GLM.
- **Tariffen beholder det meste av forbedringen.** Gini for frekvens er 0,3017, og
  halvparten av polisene får en tariffpris innenfor 5 % av GAM-prisen.
- **GLM bommer mest der modellene er uenige.** Polisene som får over 20 % lavere pris,
  hadde 17 % færre skader enn GLM forventet. De største økningene (unge førere med
  malus) ser derimot ut til å være for store i den nye tariffen.

## Begrensninger

- «Dagens tariff» er en enkel modell laget i prosjektet, ikke en ekte tariff.
  Endringene blir derfor større enn ved en vanlig revisjon.
- Datasettet har ingen årstall, så testdataene er et tilfeldig utvalg og ikke det
  siste året.
- 27 % av polisene med skademeldinger har ingen utbetaling og er regnet som skadefrie.
- Alle kurvene i hver GAM deler samme glatthetsparameter.
- Tariffen er kalibrert på historiske skader, uten justering for inflasjon eller
  fremtidig skadenivå.

## Med mer tid

- Egen glatthet per kurve, estimert med REML i mgcv (R).
- En egen indikator for bonus-malus ≥ 100, slik at hoppet der fanges av modellen.
- Undersøke samspillet mellom ung alder og malus.
- Tidsbasert split med ekte data som har årstall.

## Slik kjører du det

Krever Python 3.11 eller nyere (testet med 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
for f in code/0*.py; do python "$f"; done
```

Skriptene må kjøres i rekkefølge fra 00 til 08, siden hvert steg leser resultatene fra
de forrige. Hele kjøringen tar rundt 10 minutter og krever en del minne, mest i
steg 04.

## Merknader

- p-verdiene i pyGAMs `summary()` er upålitelige (pyGAM advarer selv om det) og brukes
  ikke i analysen.
- `code/hjelpeskript/` er ikke en del av hovedløpet:
  - `04_feilsok.py` finner ut hvorfor PoissonGAM divergerte, ved å legge til ett ledd
    om gangen
  - `04a_pyGAM_test.py` er en minimal test av PoissonGAM med eksponering
  - `05_sjekk.py` kontrollerer den flate skadestørrelsen mot testdata

## Mappestruktur

```
.
├── README.md
├── rapport.md              resultatrapport
├── requirements.txt
├── code/
│   ├── 00_lastinn.py ... 08_endringsanalyse.py
│   └── hjelpeskript/
├── data/                   lastes ned automatisk, ikke i repoet
└── results/                figurer, tabeller og JSON. Parquet- og pkl-filer lages
                            ved kjøring og er ikke i repoet
```
