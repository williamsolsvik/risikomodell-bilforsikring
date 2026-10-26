"""Minimal pyGAM-test: virker PoissonGAM med eksponering på våre data?

Ikke en del av hovedløpet. Tilpasser en liten modell (føreralder og drivstoff)
på et utvalg og sjekker at forventet antall skader treffer faktisk antall.
"""
from pathlib import Path
import numpy as np, pandas as pd
from pygam import PoissonGAM, s, f

ROOT = Path(__file__).resolve().parent.parent.parent    # prosjektmappen (mappen over code/)
OUT = ROOT / "results"

t = pd.read_parquet(OUT / "tren.parquet").sample(20_000, random_state=1)
X = np.column_stack([t["DrivAge"].to_numpy(float), t["VehGas"].cat.codes.to_numpy()])
y, e = t["N"].to_numpy(), t["Exposure"].to_numpy()

gam = PoissonGAM(s(0) + f(1)).fit(X, y, exposure=e)  # liten testmodell
gam.summary()
print("Faktisk:", y.sum(), "| Forventet:", round(gam.predict(X, exposure=e).sum(), 1))