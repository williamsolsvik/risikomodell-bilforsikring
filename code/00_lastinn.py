# 00_lastinn.py: last ned data ved behov, les ARFF, rydd datatyper, lagre som parquet
#
# Data: det franske datasettet freMTPL2 fra OpenML, freMTPL2freq (id 41214,
# poliser og eksponering) og freMTPL2sev (id 41215, skadebeløp). Mangler
# filene i data/, lastes de ned fra OpenML. md5-summen sjekkes mot den OpenML
# oppgir, slik at alle kjører på nøyaktig de samme dataene.
from pathlib import Path
import hashlib
import urllib.request
import arff            # pakken heter liac-arff, importeres som arff
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent    # prosjektmappen (mappen over code/)
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)

# Filnavn: (nedlastingsadresse, md5). Adresse og md5 er hentet fra OpenML sitt
# API, https://www.openml.org/api/v1/json/data/41214 og .../41215
KILDER = {
    "freMTPL2freq.arff": ("https://openml.org/data/v1/download/20649148/freMTPL2freq.arff",
                          "f8875568bf0ca622929105197e2db613"),
    "freMTPL2sev.arff": ("https://openml.org/data/v1/download/20649149/freMTPL2sev.arff",
                         "24cc74449e3931cb1aad0d43a12e7a6e"),
}

def md5(sti):
    with open(sti, "rb") as fil:
        return hashlib.file_digest(fil, "md5").hexdigest()

for navn, (url, forventet) in KILDER.items():
    sti = DATA / navn
    if not sti.exists():
        print(f"Laster ned {navn} fra OpenML ...")
        urllib.request.urlretrieve(url, sti)
    if md5(sti) != forventet:
        raise SystemExit(f"Feil md5 for {sti}. Slett filen og kjør skriptet på nytt.")

def les_arff(sti):
    with open(sti, "r", encoding="utf-8") as fil:
        d = arff.load(fil)                       # dict med 'attributes' og 'data'
    kolonner = [navn for navn, _ in d["attributes"]]
    df = pd.DataFrame(d["data"], columns=kolonner)
    # Tekstverdier kan ha ekstra anførselstegn, f.eks. "'Regular'"
    for k in df.select_dtypes(["object", "str"]):
        df[k] = df[k].str.strip("'\" ")
    return df

freq = les_arff(DATA / "freMTPL2freq.arff")
sev = les_arff(DATA / "freMTPL2sev.arff")

# IDpol leses som desimaltall, gjør om til heltall så koblingen blir sikker
freq["IDpol"] = freq["IDpol"].astype("int64")
sev["IDpol"] = sev["IDpol"].astype("int64")
for k in ["Area", "VehBrand", "VehGas", "Region"]:
    freq[k] = freq[k].astype("category")

print(freq.shape, sev.shape)
print(freq.dtypes)
print(freq.head())

freq.to_parquet(OUT / "freq_raa.parquet")
sev.to_parquet(OUT / "sev_raa.parquet")
print("Lagret til", OUT)