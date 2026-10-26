"""Kontroll til steg 5: treffer flat alvorlighet nivået på testdata?

Ikke en del av hovedløpet. Den flate snittskaden (kappet beløp per skade i
treningsdata, 1 715 euro) lagres av 05_alvorlighet_gam.py i results/estimater.json.
"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent    # prosjektmappen (mappen over code/)
flat = json.loads((ROOT / "results" / "estimater.json").read_text())["snittskade_flat"]  # 1715
t = pd.read_parquet(ROOT / "results" / "test.parquet")
t = t[t["n_belop"] > 0]
snitt = t["belop_kappet"].sum() / t["n_belop"].sum()
print(f"Snitt kappet beløp, test: {snitt:.0f} euro | A/E med flat {flat:,.0f} euro: {snitt / flat:.3f}")
