"""Steg 5: alvorlighets-GAM (GammaGAM på kappede snittbeløp, vektet med antall skader)."""
from pathlib import Path
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import GammaGAM, s, f
from sklearn.metrics import mean_gamma_deviance

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---- Modellvalg ----
# MED_REGION: True gir samme variabler som frekvensmodellen, False utelater Region.
MED_REGION = True
# LAM_REGEL styrer hvordan glatthetsparameteren lambda velges:
#   "min"      lambda med lavest GCV
#   "glattest" glatteste lambda innenfor 0,1 % av beste GCV
#   "fast"     lambda = 10 000, valgt fordi kurvene blir plausible, ikke ut fra GCV
LAM_REGEL = "fast"

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

tren = pd.read_parquet(OUT / "tren.parquet")
test = pd.read_parquet(OUT / "test.parquet")

KAT = ["VehBrand", "VehGas", "Region"]
KOL = ["DrivAge", "VehAge", "BonusMalus", "logDensity"] + KAT

def lag_X(df):  # identisk med steg 4
    d = pd.DataFrame(index=df.index)
    d["DrivAge"] = df["DrivAge"].clip(upper=90)
    d["VehAge"] = df["VehAge"].clip(upper=20)
    d["BonusMalus"] = df["BonusMalus"].clip(upper=150)
    d["logDensity"] = np.log(df["Density"])
    for k in KAT:
        d[k] = pd.Categorical(df[k], categories=tren[k].cat.categories).codes
    return d[KOL].to_numpy(dtype=float)

def bygg(lam):
    terms = (s(0, lam=lam) + s(1, lam=lam) + s(2, lam=lam) + s(3, lam=lam)
             + f(4, lam=lam) + f(5, lam=lam))
    if MED_REGION:
        terms = terms + f(6, lam=lam)
    return terms

# 1) Alvorlighetsdata: bare poliser med skader, snittbeløp som mål og antall skader som vekt
sev_tr, sev_te = tren[tren["n_belop"] > 0], test[test["n_belop"] > 0]
Xs_tr, Xs_te = lag_X(sev_tr), lag_X(sev_te)
ys_tr, ws_tr = sev_tr["snitt_kappet"].to_numpy(), sev_tr["n_belop"].to_numpy()
ys_te, ws_te = sev_te["snitt_kappet"].to_numpy(), sev_te["n_belop"].to_numpy()
print(f"Poliser med skader: trening {len(sev_tr)} ({ws_tr.sum()} skader), test {len(sev_te)} ({ws_te.sum()} skader)")

for j, k in zip([4, 5, 6], KAT):  # alle nivåer må ha skader, ellers kan vi ikke predikere dem
    mangler = set(range(len(tren[k].cat.categories))) - set(Xs_tr[:, j].astype(int))
    assert not mangler, f"{k}: nivåer uten skader i trening: {mangler}"

# 2) Gridsearch for glatthet med GCV (skalaen er ukjent for Gamma), ny modell per lambda
resultater = []
for lam in np.logspace(5, -1, 13):
    m = GammaGAM(bygg(lam)).fit(Xs_tr, ys_tr, weights=ws_tr)
    gcv = m.statistics_["GCV"]
    ok = np.isfinite(gcv) and np.isfinite(m.coef_).all()
    print(f"lam={lam:10.1f}  GCV={gcv:.5f}  EDoF={m.statistics_['edof']:.1f}  {'OK' if ok else 'DIVERGERTE'}")
    if ok:
        resultater.append((lam, gcv, m))
gcv_min = min(r[1] for r in resultater)
if LAM_REGEL == "min":
    beste_lam, _, gam = min(resultater, key=lambda r: r[1])
elif LAM_REGEL == "fast":  # fast lambda valgt ut fra plausibilitet, ikke ut fra GCV
    beste_lam, _, gam = next(r for r in resultater if np.isclose(r[0], 1e4))
else:  # glatteste lambda som er innenfor 0,1 % av beste GCV
    beste_lam, _, gam = max((r for r in resultater if r[1] <= gcv_min * 1.001), key=lambda r: r[0])
print("Valgt lambda:", beste_lam)
gam.summary()

# 3) Prediksjoner for ALLE poliser (trengs til risikopremien i steg 6 og 7)
lo, hi = Xs_tr[:, :4].min(axis=0), Xs_tr[:, :4].max(axis=0)
def pred(X):
    X = X.copy()
    X[:, :4] = np.clip(X[:, :4], lo, hi)  # flat kurve utenfor området vi har skadedata for
    return gam.predict(X)

pd.DataFrame({"IDpol": tren["IDpol"].to_numpy(), "sev_gam": pred(lag_X(tren))}).to_parquet(OUT / "pred_sev_tren.parquet")
pd.DataFrame({"IDpol": test["IDpol"].to_numpy(), "sev_gam": pred(lag_X(test))}).to_parquet(OUT / "pred_sev_test.parquet")
with open(OUT / "gam_alvorlighet.pkl", "wb") as fh:
    pickle.dump(gam, fh)

# 4) Fornuftssjekk mot flat baseline-alvorlighet
flat = sev_tr["belop_kappet"].sum() / ws_tr.sum()
p_tr, p_te = pred(Xs_tr), pred(Xs_te)
d_flat = mean_gamma_deviance(ys_te, np.full_like(ys_te, flat), sample_weight=ws_te)
d_gam = mean_gamma_deviance(ys_te, p_te, sample_weight=ws_te)
ae = lambda df, p: df["belop_kappet"].sum() / (df["n_belop"].to_numpy() * p).sum()
print(f"Flat baseline: {flat:.0f} euro")
print(f"A/E skadekostnad trening: {ae(sev_tr, p_tr):.3f} | test: {ae(sev_te, p_te):.3f}")
print(f"Gamma-deviance-forbedring mot flat, test: {1 - d_gam/d_flat:.2%}")

# Skalaparameteren brukes i støybåndene for kostnad i steg 6 og 8, og den flate
# snittskaden i hjelpeskript/05_sjekk.py. De avrundede verdiene er de som
# brukes, de eksakte lagres bare som dokumentasjon. Filen deles med steg 3.
EST = OUT / "estimater.json"
est = json.loads(EST.read_text()) if EST.exists() else {}
est.update(skala_sev=round(gam.distribution.scale, 1), skala_sev_eksakt=gam.distribution.scale,
           snittskade_flat=round(flat), snittskade_flat_eksakt=flat)
EST.write_text(json.dumps(est, indent=2))

# 5) Effektkurver relativt til samme referansekunde som frekvens
navn = ["Føreralder", "Bilalder", "Bonus-malus (kappet 150)", "Tetthet (log-skala)"]
ref = {0: 45, 1: 7, 2: 50, 3: np.log(300)}
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for i, ax in enumerate(axes.flat):
    XX = gam.generate_X_grid(term=i)
    kurve, ci = gam.partial_dependence(term=i, X=XX, width=0.95)
    Xr = XX[:1].copy(); Xr[0, i] = ref[i]
    niva = gam.partial_dependence(term=i, X=Xr)[0]
    x = np.exp(XX[:, i]) if i == 3 else XX[:, i]
    ax.plot(x, np.exp(kurve - niva))
    ax.fill_between(x, np.exp(ci[:, 0] - niva), np.exp(ci[:, 1] - niva), alpha=0.3)
    ax.axhline(1, color="grey", lw=0.8)
    if i == 3:
        ax.set_xscale("log")
    ax.set_title(navn[i]); ax.set_ylabel("Faktor (ref = 1)")
fig.tight_layout()
fig.savefig(OUT / "effektkurver_alvorlighet.png", dpi=120)
print("Lagret effektkurver_alvorlighet.png")