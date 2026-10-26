# 02_rens.py: bygg modelleringsdatasett for trening og test med samme regler
from pathlib import Path
import json
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
# Storskadeterskel som persentil av treningsskadene. Beløp over terskelen kappes,
# og overskuddet fordeles som et likt påslag på alle poliser. Da styres ikke
# modellene av noen få svært store skader.
TERSKEL_PCT = 0.995

freq = pd.read_parquet(OUT / "freq_raa.parquet")
sev = pd.read_parquet(OUT / "sev_raa.parquet")
tren_id = set(pd.read_parquet(OUT / "tren_id.parquet")["IDpol"])

# 1) Skaderader uten polise fjernes: de har ingen eksponering eller risikovariabler
sev = sev[sev["IDpol"].isin(freq["IDpol"])].copy()

# 2) Storskadeterskel, regnet KUN på treningsskader
sev_tren = sev.loc[sev["IDpol"].isin(tren_id), "ClaimAmount"]
print("Alternative terskler (treningsskader):")
for p in [0.99, 0.995, 0.999]:
    t = sev_tren.quantile(p)
    overskudd = (sev_tren - t).clip(lower=0).sum()
    print(f"  {p:.1%}: terskel {t:>9,.0f} euro | kappede skader {(sev_tren > t).sum():>4} | "
          f"overskudd {overskudd / sev_tren.sum():.1%} av total skadekostnad")

terskel = sev_tren.quantile(TERSKEL_PCT)
sev["belop_kappet"] = sev["ClaimAmount"].clip(upper=terskel)

# Påslag = overskudd over terskelen / sum av kappede beløp (begge på trening)
st = sev[sev["IDpol"].isin(tren_id)]
paaslag = (st["ClaimAmount"].sum() - st["belop_kappet"].sum()) / st["belop_kappet"].sum()

# 3) Summer per polise og koble på polisedata
per_polise = sev.groupby("IDpol").agg(
    n_belop=("ClaimAmount", "size"),
    belop_raa=("ClaimAmount", "sum"),
    belop_kappet=("belop_kappet", "sum"))
df = freq.join(per_polise, on="IDpol").fillna({"n_belop": 0, "belop_raa": 0.0, "belop_kappet": 0.0})
df["n_belop"] = df["n_belop"].astype(int)

# 4) Kjente feil og skadedefinisjon
df["Exposure"] = df["Exposure"].clip(upper=1)         # en polise kan ikke ha mer enn ett år
# En skade telles bare når den har et beløp i skadefilen. ClaimNb stemmer ikke
# alltid med skadefilen (se 01_utforsk.py), og slik bygger frekvens og
# alvorlighet på de samme skadene. Antallet kappes på 4 så noen få poliser med
# mange skader ikke får for stor vekt.
df["N"] = df["n_belop"].clip(upper=4)
df["snitt_kappet"] = df["belop_kappet"] / df["n_belop"].where(df["n_belop"] > 0)  # NaN uten skader
df = df.rename(columns={"ClaimNb": "ClaimNb_raa"})   # beholdes kun for sammenligning

# 5) Del i trening og test og lagre
tren = df[df["IDpol"].isin(tren_id)].copy()
test = df[~df["IDpol"].isin(tren_id)].copy()
tren.to_parquet(OUT / "tren.parquet")
test.to_parquet(OUT / "test.parquet")
with open(OUT / "storskade.json", "w") as fil:
    json.dump({"terskel_pct": TERSKEL_PCT, "terskel": float(terskel),
               "storskadepaaslag": float(paaslag)}, fil, indent=2)

# 6) Kontroller
print(f"\nValgt terskel: {terskel:,.0f} euro | storskadepåslag: {paaslag:.1%}")
print("Poliser med flere enn 4 skader med beløp:", (df["n_belop"] > 4).sum())
for navn, d in [("Trening", tren), ("Test", test)]:
    print(f"{navn}: {len(d):,} poliser | eksponering {d.Exposure.sum():,.0f} år | "
          f"frekvens ny def. {d.N.sum() / d.Exposure.sum():.4f} | "
          f"frekvens ClaimNb {d.ClaimNb_raa.clip(upper=4).sum() / d.Exposure.sum():.4f}")
b12 = tren[(tren["VehBrand"] == "B12") & (tren["VehAge"] == 0)]
print(f"B12 ny bil, frekvens ny def.: {b12.N.sum() / b12.Exposure.sum():.4f} (var 0,4528 med ClaimNb)")
print(f"Snittbeløp trening: rått {tren.belop_raa.sum() / tren.n_belop.sum():,.0f} euro | "
      f"kappet {tren.belop_kappet.sum() / tren.n_belop.sum():,.0f} euro")