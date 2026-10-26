"""Skriver ut seksjonen «Tariffen i detalj» i rapport.md som markdown.

Ikke en del av hovedløpet. Faktorene leses fra results/tariff_faktorer.csv og
grunnprisen fra results/tariff_grunnpris.json, begge laget av 07_tariff.py.
Prisen for en kunde er grunnprisen ganget med f_samlet for hver variabel, som i
07_tariff.py. Kjør etter steg 7 og lim utskriften inn nederst i rapport.md.
"""
from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent    # prosjektmappen (mappen over code/)
OUT = ROOT / "results"
fakt = pd.read_csv(OUT / "tariff_faktorer.csv", dtype={"klasse": str})
grunn = json.loads((OUT / "tariff_grunnpris.json").read_text())
GRUNNPRIS = grunn["grunnpris_med_paaslag"]

# Rekkefølge og overskrifter i rapporten
VARIABLER = {"DrivAge": "Føreralder (år)", "BonusMalus": "Bonus-malus",
             "VehAge": "Bilalder (år)", "Density": "Tetthet (innbyggere per km²)",
             "VehGas": "Drivstoff"}
SAMMENLEGGBAR = {"VehBrand": "Bilmerke", "Region": "Region"}

# Regneeksempler: klasser som avviker fra referansekunden
EKSEMPLER = [
    ("Referansekunden", {}),
    ("19 år, bonus-malus 100, bil 0–3 år, tetthet 1 001–2 000",
     {"DrivAge": "18-19", "BonusMalus": "100", "VehAge": "0-3", "Density": "1001-2000"}),
    ("30 år, bonus-malus 50, bil 15 år, tetthet 1–50",
     {"DrivAge": "30-34", "VehAge": "14-16", "Density": "1-50"}),
]


def tall(x, des=0):
    """Desimalkomma og mellomrom som tusenskille."""
    return f"{x:,.{des}f}".replace(",", " ").replace(".", ",")


def klassenavn(k):
    """18-19 blir 18–19, 1001-2000 blir 1 001–2 000, 80+ blir 80 og over."""
    if k == "Regular":
        return "Bensin (Regular)"
    if k.endswith("+"):
        return f"{tall(int(k[:-1]))} og over"
    if k.replace("-", "").isdigit():
        return "–".join(tall(int(d)) for d in k.split("-"))
    return k


def faktor(v, k):
    return fakt.loc[(fakt["variabel"] == v) & (fakt["klasse"] == k), "f_samlet"].item()


def tabell(rader):
    return "\n".join(["| Klasse | Faktor |", "| --- | --- |",
                      *(f"| {klassenavn(k)} | {tall(f, 2)} |" for k, f in rader)])


def rader(v):
    d = fakt[fakt["variabel"] == v]
    return list(zip(d["klasse"], d["f_samlet"]))


# Referanseklassen er klassen med faktor 1 for hver variabel
ref = {v: d.loc[d["f_samlet"].sub(1).abs().idxmin(), "klasse"] for v, d in fakt.groupby("variabel")}
assert all(math.isclose(faktor(v, k), 1) for v, k in ref.items())
ref_alder = grunn["referanse"].split()[0]    # 45, alderen referansekunden er definert ved

ut = [
    "## Tariffen i detalj",
    "",
    "Tariffen består av en grunnpris og en faktor per klasse for sju variabler. Prisen for "
    "en kunde er grunnprisen ganget med faktoren fra hver tabell under. Variabler som ikke "
    "er nevnt, står på referanseklassen med faktor 1.",
    "",
    f"Grunnprisen er {tall(GRUNNPRIS)} € per år. Det er prisen for referansekunden, som er "
    f"{ref_alder} år og har bil som er {klassenavn(ref['VehAge'])} år, "
    f"bonus-malus {klassenavn(ref['BonusMalus'])}, tetthet {klassenavn(ref['Density'])}, "
    f"bilmerke {ref['VehBrand']}, {klassenavn(ref['VehGas']).split()[0].lower()} "
    f"og region {ref['Region']}. Prisen er risikopremie, altså forventet skadekostnad "
    "inkludert storskadepåslaget. Påslag for kostnader og margin og justering for fremtidig "
    "skadenivå kommer i tillegg.",
    "",
    "| Kunde | Utregning | Pris per år |",
    "| --- | --- | --- |",
]
for navn, klasser in EKSEMPLER:
    f = [faktor(v, k) for v, k in klasser.items()]
    pris = GRUNNPRIS * math.prod(f)
    utregning = " x ".join([tall(GRUNNPRIS), *(tall(x, 2) for x in f)]) if f else "Grunnprisen"
    ut.append(f"| {navn} | {utregning} | {tall(pris)} € |")
ut += ["", "Prisene er regnet med uavrundede faktorer, så utregningen med de avrundede tallene "
       "kan gi et litt annet resultat.", ""]

for v, tittel in VARIABLER.items():
    ut += [f"### {tittel}", "", tabell(rader(v)), ""]

for v, tittel in SAMMENLEGGBAR.items():
    r = sorted(rader(v), key=lambda kf: int(kf[0][1:]))
    ut += ["<details>", f"<summary>{tittel} ({len(r)} klasser)</summary>", "",
           tabell(r), "", "</details>", ""]

ut.append("Faktorene delt i frekvens og skadestørrelse ligger i `results/tariff_faktorer.csv`, "
          "og grunnprisen i `results/tariff_grunnpris.json`.")
print("\n".join(ut))
