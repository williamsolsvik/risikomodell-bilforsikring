"""Steg 4: frekvens-GAM (PoissonGAM med eksponering)."""
from pathlib import Path
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pygam import PoissonGAM, s, f
from sklearn.metrics import mean_poisson_deviance

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

tren = pd.read_parquet(OUT / "tren.parquet")
test = pd.read_parquet(OUT / "test.parquet")

KAT = ["VehBrand", "VehGas", "Region"]
KOL = ["DrivAge", "VehAge", "BonusMalus", "logDensity"] + KAT  # rekkefølgen = kolonnenummer i X

def lag_X(df):
    # Kapping: det er lite data over disse grensene, og uten kapping blir kurvene
    # ustabile i halene. Tetthet er svært skjev, så den modelleres på log-skala.
    d = pd.DataFrame(index=df.index)
    d["DrivAge"] = df["DrivAge"].clip(upper=90)
    d["VehAge"] = df["VehAge"].clip(upper=20)
    d["BonusMalus"] = df["BonusMalus"].clip(upper=150)
    d["logDensity"] = np.log(df["Density"])
    for k in KAT:  # samme koding i trening og test
        d[k] = pd.Categorical(df[k], categories=tren[k].cat.categories).codes
    return d[KOL].to_numpy(dtype=float)

X_tr, X_te = lag_X(tren), lag_X(test)
assert (X_tr[:, 4:] >= 0).all() and (X_te[:, 4:] >= 0).all()  # ingen ukjente kategorier
y_tr, e_tr = tren["N"].to_numpy(), tren["Exposure"].to_numpy()
y_te, e_te = test["N"].to_numpy(), test["Exposure"].to_numpy()

# Glatte kurver for de fire kontinuerlige variablene, faktorer for de kategoriske
# (bilmerke, drivstoff og region). Alle ledd får samme glatthetsparameter lambda.
def bygg(lam):
    return (s(0, lam=lam) + s(1, lam=lam) + s(2, lam=lam) + s(3, lam=lam)
            + f(4, lam=lam) + f(5, lam=lam) + f(6, lam=lam))

# 1) Egen gridsearch på 100 000 poliser: ny modell per lambda, ingen arvet startverdi
rng = np.random.default_rng(1)
idx = rng.choice(len(X_tr), 100_000, replace=False)
beste_lam, beste_ubre = None, np.inf
for lam in np.logspace(3, -2, 11):
    m = PoissonGAM(bygg(lam)).fit(X_tr[idx], y_tr[idx], exposure=e_tr[idx])
    ubre = m.statistics_["UBRE"]
    ok = np.isfinite(ubre) and np.isfinite(m.coef_).all()
    print(f"lam={lam:9.3f}  UBRE={ubre:.5f}  EDoF={m.statistics_['edof']:.1f}  {'OK' if ok else 'DIVERGERTE'}")
    if ok and ubre < beste_ubre:
        beste_lam, beste_ubre = lam, ubre
print("Valgt lambda:", beste_lam)

# 2) Endelig modell på alle treningsdata med valgt lambda
gam = PoissonGAM(bygg(beste_lam)).fit(X_tr, y_tr, exposure=e_tr)
gam.summary()

# 3) Prediksjoner (forventet antall skader, inkl. eksponering) og lagring
pred_tr = gam.predict(X_tr, exposure=e_tr)
pred_te = gam.predict(X_te, exposure=e_te)
pd.DataFrame({"IDpol": tren["IDpol"].to_numpy(), "mu_gam": pred_tr}).to_parquet(OUT / "pred_gam_tren.parquet")
pd.DataFrame({"IDpol": test["IDpol"].to_numpy(), "mu_gam": pred_te}).to_parquet(OUT / "pred_gam_test.parquet")
with open(OUT / "gam_frekvens.pkl", "wb") as fh:
    pickle.dump(gam, fh)

# 4) Rask fornuftssjekk mot baseline (full evaluering kommer i steg 6)
glm_te = (pd.read_parquet(OUT / "pred_glm_test.parquet")
          .set_index("IDpol")["mu_glm"].reindex(test["IDpol"]).to_numpy())
null_te = e_te * y_tr.sum() / e_tr.sum()  # nullmodell: én frekvens for alle
d_null, d_glm, d_gam = (mean_poisson_deviance(y_te, p) for p in (null_te, glm_te, pred_te))
print(f"A/E trening: {y_tr.sum()/pred_tr.sum():.3f} | A/E test: {y_te.sum()/pred_te.sum():.3f}")
print(f"Deviance-forbedring mot nullmodell, test: GLM {1-d_glm/d_null:.2%} | GAM {1-d_gam/d_null:.2%}")

# 5) Effektkurver som faktorer relativt til samme referansekunde som baseline
navn = ["Føreralder", "Bilalder", "Bonus-malus (kappet 150)", "Tetthet (log-skala)"]
ref = {0: 45, 1: 7, 2: 50, 3: np.log(300)}
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for i, ax in enumerate(axes.flat):
    XX = gam.generate_X_grid(term=i)
    kurve, ci = gam.partial_dependence(term=i, X=XX, width=0.95)
    Xr = XX[:1].copy(); Xr[0, i] = ref[i]
    niva = gam.partial_dependence(term=i, X=Xr)[0]   # kurvens verdi i referansepunktet
    x = np.exp(XX[:, i]) if i == 3 else XX[:, i]
    ax.plot(x, np.exp(kurve - niva))
    ax.fill_between(x, np.exp(ci[:, 0] - niva), np.exp(ci[:, 1] - niva), alpha=0.3)
    ax.axhline(1, color="grey", lw=0.8)
    if i == 3:
        ax.set_xscale("log")
    ax.set_title(navn[i]); ax.set_ylabel("Faktor (ref = 1)")
fig.tight_layout()
fig.savefig(OUT / "effektkurver_frekvens.png", dpi=120)
print("Lagret effektkurver_frekvens.png")