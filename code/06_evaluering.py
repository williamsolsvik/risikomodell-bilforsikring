"""
Steg 6: Evaluering på testdata.
Sammenligner GAM (frekvens-GAM x alvorlighets-GAM) mot baseline
(GLM x flat alvorlighet).

Alt sammenlignes UTEN storskadepåslag: kappet faktisk kostnad mot kappet
forventet kostnad. Påslaget er likt for alle og endrer ikke rangeringen.

Rekkefølge:
  1. Totalnivå (A/E), deretter normaliseres begge modellene til totalen
  2. A/E per segment (frekvens)
  3. Lift og double lift (frekvens og kostnad)
  4. Gini med bootstrap (er forskjellen reell?)
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
OVERDISP = est["overdisp"]                 # 1.72, overdispersjon fra baseline-GLM (03_baseline_glm.py)
SKALA_SEV = est["skala_sev"]               # 1.9, skalaparameter fra alvorlighets-GAM (05_alvorlighet_gam.py)
STOY_FREKV = OVERDISP                      # støyfaktor for antall skader
STOY_KOST = OVERDISP * (1 + SKALA_SEV)     # grovt anslag: antallsstøy + beløpsstøy
N_BOOT = 200

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 50)
pd.set_option("display.float_format",
              lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.0f}")

# ---------------------------------------------------------------------------
# 0. Data
# ---------------------------------------------------------------------------
tren = pd.read_parquet(OUT / "tren.parquet", columns=["belop_kappet", "n_belop"])
test = pd.read_parquet(OUT / "test.parquet")
n_test = len(test)
for navn in ["pred_glm_test", "pred_gam_test", "pred_sev_test"]:
    test = test.merge(pd.read_parquet(OUT / f"{navn}.parquet"),
                      on="IDpol", how="left", validate="1:1")
assert len(test) == n_test, "Antall rader endret seg i merge"
assert test[["mu_glm", "mu_gam", "sev_gam"]].notna().all().all(), \
    "Mangler prediksjoner for noen poliser"

sev_flat = tren["belop_kappet"].fillna(0).sum() / tren["n_belop"].fillna(0).sum()
print(f"Flat alvorlighet fra trening: {sev_flat:,.0f} euro "
      f"(steg 5 fant {est['snittskade_flat']:,.0f} euro)")

test["kost"] = test["belop_kappet"].fillna(0)
test["rp_glm"] = test["mu_glm"] * sev_flat              # forventet kappet kostnad
test["rp_gam"] = test["mu_gam"] * test["sev_gam"]

# ---------------------------------------------------------------------------
# 1. Totalnivå
# ---------------------------------------------------------------------------
print("\n=== 1. Totalnivå på test (faktisk / forventet) ===")
for m in ["glm", "gam"]:
    ae_f = test["N"].sum() / test[f"mu_{m}"].sum()
    ae_k = test["kost"].sum() / test[f"rp_{m}"].sum()
    print(f"{m.upper()}: frekvens {ae_f:.3f}, kostnad {ae_k:.3f}")
print(f"Antall skader på test: {test['N'].sum():,.0f}")

# Normaliser: begge modellene skaleres til å treffe totalen på test.
# Da måler resten bare hvordan modellene FORDELER risikoen, ikke nivået.
# Nivået håndteres i kalibreringen i steg 7.
for m in ["glm", "gam"]:
    test[f"mu_{m}_n"] = test[f"mu_{m}"] * test["N"].sum() / test[f"mu_{m}"].sum()
    test[f"rp_{m}_n"] = test[f"rp_{m}"] * test["kost"].sum() / test[f"rp_{m}"].sum()

# ---------------------------------------------------------------------------
# 2. A/E per segment (frekvens, normalisert)
# ---------------------------------------------------------------------------
test["Alder"] = pd.cut(
    test["DrivAge"], [17, 19, 21, 24, 29, 39, 44, 49, 54, 59, 69, 200],
    labels=["18-19", "20-21", "22-24", "25-29", "30-39", "40-44",
            "45-49", "50-54", "55-59", "60-69", "70+"])
test["Bilalder"] = pd.cut(
    test["VehAge"], [-1, 0, 3, 7, 12, 19, 200],
    labels=["0", "1-3", "4-7", "8-12", "13-19", "20+"])
test["BM_gruppe"] = pd.cut(
    test["BonusMalus"], [49, 50, 59, 79, 89, 99, 100, 110, 1000],
    labels=["50", "51-59", "60-79", "80-89", "90-99", "100", "101-110", "111+"])


def ae_segment(df, gruppe):
    g = df.groupby(gruppe, observed=True).agg(
        eksp=("Exposure", "sum"), skader=("N", "sum"),
        forv_glm=("mu_glm_n", "sum"), forv_gam=("mu_gam_n", "sum"))
    g["AE_glm"] = g["skader"] / g["forv_glm"]
    g["AE_gam"] = g["skader"] / g["forv_gam"]
    g["stoy_pm"] = 2 * np.sqrt(STOY_FREKV / g["forv_gam"])   # grovt 95 %-bånd
    return g[["eksp", "skader", "AE_glm", "AE_gam", "stoy_pm"]]


def snittavvik(g):
    w = g["eksp"] / g["eksp"].sum()
    return (w * (g["AE_glm"] - 1).abs()).sum(), (w * (g["AE_gam"] - 1).abs()).sum()


print("\n=== 2. A/E per segment, frekvens (normalisert, 1 = treffer som totalen) ===")
print("stoy_pm: avvik fra 1 innenfor ± dette er ikke et signal")
segmenter = []
for var in ["Alder", "Bilalder", "BM_gruppe", "Region"]:
    g = ae_segment(test, var)
    if var == "Region":
        g = g.sort_values("eksp", ascending=False)
    a_glm, a_gam = snittavvik(g)
    print(f"\n--- {var} ---")
    print(g)
    print(f"Snittavvik fra 1 (vektet med eksponering): GLM {a_glm:.3f}, GAM {a_gam:.3f}")
    segmenter.append(g.reset_index().rename(columns={var: "niva"}).assign(variabel=var))
pd.concat(segmenter).to_csv(OUT / "evaluering_ae_segment.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Lift og double lift
# ---------------------------------------------------------------------------
def grupper_etter(df, kolonne, n):
    """Sorterer etter kolonne og deler i n grupper med lik eksponering."""
    d = df.sort_values(kolonne, kind="mergesort")
    andel = d["Exposure"].cumsum() / d["Exposure"].sum()
    d["gruppe"] = np.clip(np.ceil(andel * n), 1, n).astype(int)
    return d


def lift(df, forv, faktisk, n):
    d = grupper_etter(df.assign(rate=df[forv] / df["Exposure"]), "rate", n)
    g = d.groupby("gruppe").agg(eksp=("Exposure", "sum"),
                                forv=(forv, "sum"), faktisk=(faktisk, "sum"))
    g["forv_per_aar"] = g["forv"] / g["eksp"]
    g["faktisk_per_aar"] = g["faktisk"] / g["eksp"]
    return g


def lift_tabell(forv_glm, forv_gam, faktisk, n):
    lg, la = lift(test, forv_glm, faktisk, n), lift(test, forv_gam, faktisk, n)
    t = pd.DataFrame({"GLM_forv": lg["forv_per_aar"], "GLM_faktisk": lg["faktisk_per_aar"],
                      "GAM_forv": la["forv_per_aar"], "GAM_faktisk": la["faktisk_per_aar"]})
    return t


def double_lift(df, ny, gammel, faktisk, n, stoyfaktor):
    """Sorterer etter forholdet GAM/GLM. Der modellene er uenige: hvem har rett?"""
    d = grupper_etter(df.assign(forhold=df[ny] / df[gammel]), "forhold", n)
    g = d.groupby("gruppe").agg(eksp=("Exposure", "sum"), faktisk=(faktisk, "sum"),
                                gammel=(gammel, "sum"), ny=(ny, "sum"),
                                forv_skader=("mu_gam_n", "sum"))
    g["GAM/GLM"] = g["ny"] / g["gammel"]
    g["AE_glm"] = g["faktisk"] / g["gammel"]
    g["AE_gam"] = g["faktisk"] / g["ny"]
    g["stoy_pm"] = 2 * np.sqrt(stoyfaktor / g["forv_skader"])
    return g[["eksp", "GAM/GLM", "AE_glm", "AE_gam", "stoy_pm"]]


print("\n=== 3a. Lift, frekvens (10 grupper, skader per eksponeringsår) ===")
lift_f = lift_tabell("mu_glm_n", "mu_gam_n", "N", 10)
print(lift_f)
print(f"Faktisk frekvens høyeste/laveste gruppe: "
      f"GLM {lift_f['GLM_faktisk'].iloc[-1] / lift_f['GLM_faktisk'].iloc[0]:.1f}x, "
      f"GAM {lift_f['GAM_faktisk'].iloc[-1] / lift_f['GAM_faktisk'].iloc[0]:.1f}x")

print("\n=== 3b. Lift, kostnad (5 grupper, euro per eksponeringsår) ===")
lift_k = lift_tabell("rp_glm_n", "rp_gam_n", "kost", 5)
print(lift_k)
print(f"Faktisk kostnad høyeste/laveste gruppe: "
      f"GLM {lift_k['GLM_faktisk'].iloc[-1] / lift_k['GLM_faktisk'].iloc[0]:.1f}x, "
      f"GAM {lift_k['GAM_faktisk'].iloc[-1] / lift_k['GAM_faktisk'].iloc[0]:.1f}x")

print("\n=== 3c. Double lift, frekvens (10 grupper, sortert etter GAM/GLM) ===")
dl_f = double_lift(test, "mu_gam_n", "mu_glm_n", "N", 10, STOY_FREKV)
print(dl_f)

print("\n=== 3d. Double lift, kostnad (5 grupper, sortert etter GAM/GLM) ===")
dl_k = double_lift(test, "rp_gam_n", "rp_glm_n", "kost", 5, STOY_KOST)
print(dl_k)

# ---------------------------------------------------------------------------
# 4. Gini med bootstrap
# ---------------------------------------------------------------------------
def gini(faktisk, forv, eksp):
    """Hvor godt rangerer modellen? 0 = tilfeldig, høyere = bedre."""
    rekkef = np.argsort(forv / eksp, kind="mergesort")
    x = np.r_[0, np.cumsum(eksp[rekkef])] / eksp.sum()
    y = np.r_[0, np.cumsum(faktisk[rekkef])] / faktisk.sum()
    return 1 - 2 * np.trapezoid(y, x)


print(f"\n=== 4. Gini på test, med {N_BOOT} bootstrap-trekk (tar litt tid) ===")
rng = np.random.default_rng(1)
eksp = test["Exposure"].to_numpy()
for maal, fakt, glm, gam in [("frekvens", "N", "mu_glm", "mu_gam"),
                             ("kostnad", "kost", "rp_glm", "rp_gam")]:
    y, p_glm, p_gam = (test[c].to_numpy() for c in (fakt, glm, gam))
    g_glm, g_gam = gini(y, p_glm, eksp), gini(y, p_gam, eksp)
    diff = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(eksp), len(eksp))
        diff.append(gini(y[i], p_gam[i], eksp[i]) - gini(y[i], p_glm[i], eksp[i]))
    lav, hoy = np.percentile(diff, [2.5, 97.5])
    andel_pos = np.mean(np.array(diff) > 0)
    print(f"{maal}: GLM {g_glm:.4f}, GAM {g_gam:.4f}, forskjell {g_gam - g_glm:+.4f} "
          f"(95 %-intervall {lav:+.4f} til {hoy:+.4f}, GAM best i {andel_pos:.0%} av trekkene)")

# ---------------------------------------------------------------------------
# 5. Plott
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for a, t, tittel, enhet in [(ax[0, 0], lift_f, "Lift, frekvens", "skader per år"),
                            (ax[1, 0], lift_k, "Lift, kostnad", "euro per år")]:
    x = t.index
    a.plot(x, t["GLM_forv"], "-", color="tab:blue", label="GLM forventet")
    a.plot(x, t["GLM_faktisk"], "o", color="tab:blue", label="GLM faktisk")
    a.plot(x, t["GAM_forv"], "-", color="tab:orange", label="GAM forventet")
    a.plot(x, t["GAM_faktisk"], "s", color="tab:orange", label="GAM faktisk")
    a.set_title(tittel), a.set_xlabel("gruppe (lav til høy forventet risiko)")
    a.set_ylabel(enhet), a.legend(fontsize=8)

for a, t, tittel in [(ax[0, 1], dl_f, "Double lift, frekvens"),
                     (ax[1, 1], dl_k, "Double lift, kostnad")]:
    x = t.index
    a.fill_between(x, 1 - t["stoy_pm"], 1 + t["stoy_pm"], color="grey", alpha=0.2,
                   label="støybånd")
    a.axhline(1, color="black", lw=0.8)
    a.plot(x, t["AE_glm"], "o-", color="tab:blue", label="A/E GLM")
    a.plot(x, t["AE_gam"], "s-", color="tab:orange", label="A/E GAM")
    a.set_title(tittel), a.set_xlabel("gruppe (GAM lavere enn GLM → GAM høyere)")
    a.set_ylabel("faktisk / forventet"), a.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "evaluering_lift.png", dpi=120)
print(f"\nLagret: {OUT / 'evaluering_lift.png'} og evaluering_ae_segment.csv")