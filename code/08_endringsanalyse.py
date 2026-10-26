"""
Steg 8: Endringsanalyse.
Ny tariff (steg 7) mot baseline (GLM x flat alvorlighet) for hver polise.

1. Nivå: begge tariffene på samme totale premie (inntektsnøytralt)
2. Fordeling av endringene: hvor mange får store endringer?
3. Endring per segment: hvem får det?
4. De største endringene: hvordan ser de kundene ut?
5. Test: for kundene som får store endringer, hvem av tariffene har rett?

Premie = per eksponeringsår, med storskadepåslag (det kunden ser).
Endringsanalysen bruker alle poliser, fordi den bare bruker priser, ikke skader.
Bare del 5 bruker skadene, og bare på testdata.
"""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
# Avrundede estimater fra tidligere steg, lagret i results/estimater.json
est = json.loads((OUT / "estimater.json").read_text())
OVERDISP = est["overdisp"]     # 1.72, overdispersjon fra baseline-GLM (03_baseline_glm.py)
SKALA_SEV = est["skala_sev"]   # 1.9, skalaparameter fra alvorlighets-GAM (05_alvorlighet_gam.py)
STOY_KOST = OVERDISP * (1 + SKALA_SEV)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 60)
pd.set_option("display.float_format",
              lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.0f}")

# Samme klasser som i steg 7
GRENSER = {
    "DrivAge": [18, 20, 22, 24, 26, 30, 35, 40, 45, 50, 55, 60, 70, 80],
    "VehAge": [0, 4, 8, 11, 14, 17, 20],
    "BonusMalus": [50, 51, 56, 61, 71, 81, 91, 100, 101, 111, 126],
    "Density": [1, 51, 101, 201, 501, 1001, 2001, 5001, 10001],
}
BOTTER = [0, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, np.inf]
BOTTE_NAVN = ["ned over 20 %", "ned 10-20 %", "ned 5-10 %", "innenfor ±5 %",
              "opp 5-10 %", "opp 10-20 %", "opp over 20 %"]

# ---------------------------------------------------------------------------
# 0. Data
# ---------------------------------------------------------------------------
def last(navn):
    df = pd.read_parquet(OUT / f"{navn}.parquet")
    for p in [f"pred_glm_{navn}", f"pred_tariff_{navn}"]:
        df = df.merge(pd.read_parquet(OUT / f"{p}.parquet"), on="IDpol", validate="1:1")
    return df.assign(sett=navn)


alle = pd.concat([last("tren"), last("test")], ignore_index=True)
for c in ["VehBrand", "VehGas", "Region"]:
    alle[c] = alle[c].astype(str)
alle["kost"] = alle["belop_kappet"].fillna(0)

with open(OUT / "storskade.json") as fh:
    storskade = json.load(fh)
PAASLAG = next(v for k, v in storskade.items() if "slag" in k)

tren = alle[alle["sett"] == "tren"]
sev_flat = tren["belop_kappet"].fillna(0).sum() / tren["n_belop"].fillna(0).sum()

alle["gammel"] = alle["mu_glm"] / alle["Exposure"] * sev_flat * (1 + PAASLAG)
alle["ny_raa"] = alle["rp_tariff"] / alle["Exposure"] * (1 + PAASLAG)
skal = (alle["gammel"] * alle["Exposure"]).sum() / (alle["ny_raa"] * alle["Exposure"]).sum()
alle["ny"] = alle["ny_raa"] * skal
alle["forhold"] = alle["ny"] / alle["gammel"]
alle["endring_kr"] = alle["ny"] - alle["gammel"]

# ---------------------------------------------------------------------------
# 1. Nivå
# ---------------------------------------------------------------------------
print("=== 1. Nivå (per eksponeringsår, med påslag) ===")
snitt = lambda kol: (alle[kol] * alle["Exposure"]).sum() / alle["Exposure"].sum()
print(f"Snittpremie gammel {snitt('gammel'):,.0f} euro, ny før skalering {snitt('ny_raa'):,.0f} euro")
print(f"Skaleringsfaktor for inntektsnøytralitet: {skal:.4f} (nær 1 fordi begge er kalibrert på trening)")
print(f"Antall poliser: {len(alle):,}")

# ---------------------------------------------------------------------------
# 2. Fordeling av endringene
# ---------------------------------------------------------------------------
print("\n=== 2. Fordeling av endringene (ny / gammel per polise) ===")
pct = np.percentile(alle["forhold"], [1, 5, 10, 25, 50, 75, 90, 95, 99])
print("Persentiler 1, 5, 10, 25, 50, 75, 90, 95, 99:")
print("  " + "  ".join(f"{p:.3f}" for p in pct))
print(f"Snitt absolutt endring: {alle['endring_kr'].abs().mean():,.0f} euro per år")

alle["botte"] = pd.cut(alle["forhold"], BOTTER, labels=BOTTE_NAVN, right=False)
b = alle.groupby("botte", observed=False).agg(
    poliser=("IDpol", "size"), eksp=("Exposure", "sum"),
    gammel=("gammel", "mean"), endring_kr=("endring_kr", "mean"))
b["andel_poliser"] = b["poliser"] / b["poliser"].sum()
print(b[["poliser", "andel_poliser", "gammel", "endring_kr"]])
for t in (0.10, 0.15, 0.20):
    print(f"Et tak på ±{t:.0%} ville berørt {(np.abs(alle['forhold'] - 1) > t).mean():.1%} av polisene")

# ---------------------------------------------------------------------------
# 3. Endring per segment
# ---------------------------------------------------------------------------
def etiketter(grenser):
    ut = []
    for i, lo in enumerate(grenser):
        if i + 1 < len(grenser):
            hi = grenser[i + 1] - 1
            ut.append(str(lo) if hi == lo else f"{lo}-{hi}")
        else:
            ut.append(f"{lo}+")
    return ut


def klasse(df, v):
    if v in GRENSER:
        g = GRENSER[v]
        lab = etiketter(g)
        i = np.searchsorted(np.asarray(g), df[v].to_numpy(), side="right") - 1
        return pd.Categorical(np.array(lab)[i], categories=lab, ordered=True)
    return df[v]


def segment(v):
    d = alle.assign(k=klasse(alle, v), gp=alle["gammel"] * alle["Exposure"],
                    nyp=alle["ny"] * alle["Exposure"],
                    opp20=alle["forhold"] >= 1.2, ned20=alle["forhold"] < 0.8)
    g = d.groupby("k", observed=True).agg(
        poliser=("IDpol", "size"), eksp=("Exposure", "sum"), gp=("gp", "sum"),
        nyp=("nyp", "sum"), opp20=("opp20", "mean"), ned20=("ned20", "mean"))
    g["gammel"] = g["gp"] / g["eksp"]
    g["ny"] = g["nyp"] / g["eksp"]
    g["endring"] = g["ny"] / g["gammel"] - 1
    g = g[["poliser", "gammel", "ny", "endring", "opp20", "ned20"]]
    if v in ("VehBrand", "Region"):
        g = g.sort_values("endring")
    return g


print("\n=== 3. Endring per segment ===")
print("gammel/ny: snittpremie euro per år. endring, opp20, ned20: andeler (0,05 = 5 %)")
print("opp20/ned20: andel av polisene i segmentet som får over 20 % opp/ned")
seg_tab = []
for v in ["DrivAge", "BonusMalus", "VehAge", "Density", "VehGas", "VehBrand", "Region"]:
    g = segment(v)
    print(f"\n--- {v} ---")
    print(g)
    seg_tab.append(g.reset_index().rename(columns={"k": "klasse"}).assign(variabel=v))
pd.concat(seg_tab).to_csv(OUT / "endringsanalyse_segment.csv", index=False)

# ---------------------------------------------------------------------------
# 4. De største endringene
# ---------------------------------------------------------------------------
kol = ["DrivAge", "VehAge", "BonusMalus", "Density", "VehBrand", "VehGas", "Region",
       "gammel", "ny", "forhold"]
print("\n=== 4a. Største økninger ===")
print(alle.nlargest(10, "forhold")[kol].to_string(index=False))
print("\n=== 4b. Største reduksjoner ===")
print(alle.nsmallest(10, "forhold")[kol].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Test: hvem har rett for kundene som får store endringer?
# ---------------------------------------------------------------------------
t = alle[alle["sett"] == "test"].copy()
t["forv_kost_g"] = t["gammel"] * t["Exposure"]
t["forv_kost_n"] = t["ny"] * t["Exposure"]
for c, fakt in [("forv_kost_g", "kost"), ("forv_kost_n", "kost"),
                ("mu_glm", "N"), ("forv_N", "N")]:
    t[c] = t[c] * t[fakt].sum() / t[c].sum()       # normalisert til totalen på test

g = t.groupby("botte", observed=False).agg(
    poliser=("IDpol", "size"), kost=("kost", "sum"), N=("N", "sum"),
    kg=("forv_kost_g", "sum"), kn=("forv_kost_n", "sum"),
    fg=("mu_glm", "sum"), fn=("forv_N", "sum"))
g["AE_kost_gammel"] = g["kost"] / g["kg"]
g["AE_kost_ny"] = g["kost"] / g["kn"]
g["stoy_kost"] = 2 * np.sqrt(STOY_KOST / g["fn"])
g["AE_frekv_gammel"] = g["N"] / g["fg"]
g["AE_frekv_ny"] = g["N"] / g["fn"]
g["stoy_frekv"] = 2 * np.sqrt(OVERDISP / g["fn"])
print("\n=== 5. Test: faktisk / forventet per endringsgruppe (normalisert, 1 = som totalen) ===")
print(g[["poliser", "N", "AE_kost_gammel", "AE_kost_ny", "stoy_kost",
         "AE_frekv_gammel", "AE_frekv_ny", "stoy_frekv"]])

# ---------------------------------------------------------------------------
# 6. Histogram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(alle["forhold"], bins=np.geomspace(0.5, 2, 61), color="tab:blue")
ax.set_xscale("log")
ax.set_xticks([0.5, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.5, 2])
ax.set_xticklabels(["0,5", "0,7", "0,8", "0,9", "1", "1,1", "1,2", "1,5", "2"])
for x in (0.8, 1.2):
    ax.axvline(x, color="black", lw=0.8, ls="--")
ax.set_xlabel("ny premie / gammel premie")
ax.set_ylabel("antall poliser")
ax.set_title("Endring i premie per polise")
fig.tight_layout()
fig.savefig(OUT / "endringsanalyse.png", dpi=120)
print("\nLagret: endringsanalyse_segment.csv og endringsanalyse.png")