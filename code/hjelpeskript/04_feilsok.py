"""Feilsøking til steg 4: hvorfor divergerer PoissonGAM?

Ikke en del av hovedløpet. Legger til ett ledd om gangen på et utvalg av
treningsdata og sjekker om modellen konvergerer, og prøver deretter full modell
med svært høy og svært lav lambda.
"""
from pathlib import Path
import warnings
import numpy as np, pandas as pd
from pygam import PoissonGAM, s, f
warnings.filterwarnings("ignore"); np.seterr(all="ignore")  # rydder bort støyen

ROOT = Path(__file__).resolve().parent.parent.parent    # prosjektmappen (mappen over code/)
OUT = ROOT / "results"
t = pd.read_parquet(OUT / "tren.parquet").sample(20_000, random_state=1)
X = np.column_stack([t["DrivAge"].clip(upper=90), t["VehAge"].clip(upper=20),
                     t["BonusMalus"].clip(upper=150), np.log(t["Density"]),
                     t["VehBrand"].cat.codes, t["VehGas"].cat.codes, t["Region"].cat.codes]).astype(float)
y, e = t["N"].to_numpy(), t["Exposure"].to_numpy()

def bygg(antall, lam):  # de første 'antall' leddene, alle med samme lambda
    ledd = [s(0, lam=lam), s(1, lam=lam), s(2, lam=lam), s(3, lam=lam),
            f(4, lam=lam), f(5, lam=lam), f(6, lam=lam)][:antall]
    terms = ledd[0]
    for l in ledd[1:]:
        terms = terms + l
    return terms

def prov(antall, lam, navn):  # ny modell hver gang, ingen arvet startverdi
    p = PoissonGAM(bygg(antall, lam)).fit(X, y, exposure=e).predict(X, exposure=e)
    ok = np.isfinite(p).all() and abs(y.sum() / p.sum() - 1) < 0.01
    print(f"{navn:22s} lam={lam:<6} {'OK' if ok else 'DIVERGERTE'}")

for antall, navn in enumerate(["føreralder", "+ bilalder", "+ bonus-malus", "+ tetthet",
                               "+ bilmerke", "+ drivstoff", "+ region"], start=1):
    prov(antall, 0.6, navn)
prov(7, 1000, "full modell")
prov(7, 0.001, "full modell")