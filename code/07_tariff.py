"""
Steg 7: Fra glatte GAM-kurver til tariff.

1. Sjekk at variabelforberedelsen gir samme prediksjoner som i steg 4 og 5
2. Del hver kontinuerlige variabel i tariffklasser
3. Faktor per klasse = snittet av GAM-kurven innenfor klassen (vektet med
   eksponering), for frekvens og alvorlighet hver for seg, så ganget sammen
4. Kalibrer grunnprisen på treningsdata, legg på storskadepåslaget
5. Sjekk på trening: A/E per klasse (her kan vi justere)
6. Sjekk på test (bare én gang): hvor mye tapte vi på klasseinndelingen?
"""
from pathlib import Path
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OVERDISP = json.loads((OUT / "estimater.json").read_text())["overdisp"]  # 1.72, fra 03_baseline_glm.py

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 60)
pd.set_option("display.float_format",
              lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.0f}")

# ---------------------------------------------------------------------------
# Tariffklasser (nedre grense for hver klasse)
# ---------------------------------------------------------------------------
KONT = {
    "DrivAge": dict(kol=0, grenser=[18, 20, 22, 24, 26, 30, 35, 40, 45, 50, 55, 60, 70, 80]),
    # Bilalder: frekvens flat 0-7 (svingningene var støy), alvorlighet får 0-3 og 4-7
    "VehAge": dict(kol=1, grenser=[0, 4, 8, 11, 14, 17, 20],
                   grenser_frekv=[0, 8, 11, 14, 17, 20]),
    # Bonus-malus: egen klasse for 100, slik at tariffen kan ha et hopp der
    "BonusMalus": dict(kol=2, grenser=[50, 51, 56, 61, 71, 81, 91, 100, 101, 111, 126]),
    # Tetthet: kurven holdes flat under 10 (randeffekt med svært bredt bånd)
    "Density": dict(kol=3, grenser=[1, 51, 101, 201, 501, 1001, 2001, 5001, 10001]),
}
KAT = {"VehBrand": 4, "VehGas": 5, "Region": 6}
REF = {"DrivAge": 45, "VehAge": 7, "BonusMalus": 50, "Density": 300}
TETTHET_FLAT_UNDER = 10

# ---------------------------------------------------------------------------
# 0. Data og modeller
# ---------------------------------------------------------------------------
tren = (pd.read_parquet(OUT / "tren.parquet")
        .merge(pd.read_parquet(OUT / "pred_gam_tren.parquet"), on="IDpol", validate="1:1")
        .merge(pd.read_parquet(OUT / "pred_sev_tren.parquet"), on="IDpol", validate="1:1"))
test = (pd.read_parquet(OUT / "test.parquet")
        .merge(pd.read_parquet(OUT / "pred_gam_test.parquet"), on="IDpol", validate="1:1")
        .merge(pd.read_parquet(OUT / "pred_sev_test.parquet"), on="IDpol", validate="1:1"))
for df in (tren, test):
    df["kost"] = df["belop_kappet"].fillna(0)
    df["rp_gam"] = df["mu_gam"] * df["sev_gam"]

with open(OUT / "gam_frekvens.pkl", "rb") as fh:
    gam_f = pickle.load(fh)
with open(OUT / "gam_alvorlighet.pkl", "rb") as fh:
    gam_s = pickle.load(fh)
with open(OUT / "storskade.json") as fh:
    storskade = json.load(fh)
PAASLAG = next(v for k, v in storskade.items() if "slag" in k)
print(f"Storskadepåslag: {PAASLAG:.3f}")

KAT_NIVAA = {c: tren[c].cat.categories for c in KAT}


def lag_X(df):
    """Samme forberedelse som i steg 4 og 5."""
    return np.column_stack([
        df["DrivAge"].clip(upper=90),
        df["VehAge"].clip(upper=20),
        df["BonusMalus"].clip(upper=150),
        np.log(df["Density"]),
        *[pd.Categorical(df[c], categories=KAT_NIVAA[c]).codes for c in KAT],
    ]).astype(float)


X_tren = lag_X(tren)
med_skade = tren["n_belop"].fillna(0).to_numpy() > 0
SEV_MIN = X_tren[med_skade, :4].min(axis=0)
SEV_MAX = X_tren[med_skade, :4].max(axis=0)


def klipp_sev(X):
    """Alvorlighet holdes flat utenfor området med skadedata (som i steg 5)."""
    X = X.copy()
    X[:, :4] = np.clip(X[:, :4], SEV_MIN, SEV_MAX)
    return X


def klipp_flat(X):
    """Tetthet under 10 får samme verdi som tetthet 10."""
    X = X.copy()
    X[:, 3] = np.maximum(X[:, 3], np.log(TETTHET_FLAT_UNDER))
    return X


# ---------------------------------------------------------------------------
# 1. Sikkerhetssjekk
# ---------------------------------------------------------------------------
print("\n=== 1. Sjekk mot prediksjonene fra steg 4 og 5 ===")
mu_sjekk = gam_f.predict(X_tren) * tren["Exposure"].to_numpy()
avvik_f = np.max(np.abs(mu_sjekk / tren["mu_gam"].to_numpy() - 1))
sev_sjekk = gam_s.predict(klipp_sev(X_tren))
avvik_s = np.max(np.abs(sev_sjekk / tren["sev_gam"].to_numpy() - 1))
print(f"Største relative avvik: frekvens {avvik_f:.2e}, alvorlighet {avvik_s:.2e}")
assert avvik_f < 1e-6, "Frekvens avviker fra steg 4. Kjør steg 4 på nytt."
if avvik_s > 1e-6:
    print("ADVARSEL: alvorlighet avviker fra steg 5. Kjør steg 5 på nytt før du går videre.")

for navn, gam, X in [("frekvens", gam_f, X_tren), ("alvorlighet", gam_s, klipp_sev(X_tren))]:
    sum_ledd = sum(np.ravel(gam.partial_dependence(term=j, X=X)) for j in range(7))
    rest = np.log(gam.predict(X)) - sum_ledd
    print(f"{navn}: leddene summerer til modellen (spredning i resten {np.std(rest):.1e}, skal være ~0)")

# ---------------------------------------------------------------------------
# 2-3. Klassefaktorer
# ---------------------------------------------------------------------------
def klasse_idx(x, grenser):
    i = np.searchsorted(np.asarray(grenser), np.asarray(x), side="right") - 1
    assert (i >= 0).all(), "Verdi under laveste klassegrense"
    return i


def etiketter(grenser):
    ut = []
    for i, lo in enumerate(grenser):
        if i + 1 < len(grenser):
            hi = grenser[i + 1] - 1
            ut.append(str(lo) if hi == lo else f"{lo}-{hi}")
        else:
            ut.append(f"{lo}+")
    return ut


def klassefaktorer(kurve, idx, w, ref, n):
    """Snitt av kurven (log-skala) per klasse, vektet med eksponering, relativt til ref."""
    sum_w = np.bincount(idx, weights=w, minlength=n)
    assert (sum_w > 0).all(), "En klasse har ingen eksponering"
    log_snitt = np.bincount(idx, weights=w * kurve, minlength=n) / sum_w
    return np.exp(log_snitt - log_snitt[ref]), log_snitt[ref]


eksp = tren["Exposure"].to_numpy()
MODELL = {"frekv": (gam_f, klipp_flat(X_tren)),
          "alv": (gam_s, klipp_sev(klipp_flat(X_tren)))}
FAKT = {}
for komp, (gam, X) in MODELL.items():
    for v, cfg in KONT.items():
        g = cfg.get(f"grenser_{komp}", cfg["grenser"])
        idx = klasse_idx(tren[v], g)
        ref = klasse_idx([REF[v]], g)[0]
        kurve = np.ravel(gam.partial_dependence(term=cfg["kol"], X=X))
        f, logref = klassefaktorer(kurve, idx, eksp, ref, len(g))
        FAKT[(v, komp)] = dict(grenser=g, f=f, logref=logref)
    for v, kol in KAT.items():
        idx = X_tren[:, kol].astype(int)
        n = len(KAT_NIVAA[v])
        ref = int(np.argmax(np.bincount(idx, weights=eksp, minlength=n)))
        kurve = np.ravel(gam.partial_dependence(term=kol, X=X))
        f, logref = klassefaktorer(kurve, idx, eksp, ref, n)
        FAKT[(v, komp)] = dict(f=f, logref=logref)


def policefaktor(df, v, komp):
    F = FAKT[(v, komp)]
    if v in KONT:
        return F["f"][klasse_idx(df[v], F["grenser"])]
    return F["f"][pd.Categorical(df[v], categories=KAT_NIVAA[v]).codes]


def relativ(df, komp):
    return np.prod([policefaktor(df, v, komp) for v in [*KONT, *KAT]], axis=0)


# ---------------------------------------------------------------------------
# 4. Kalibrering av grunnpris (på treningsdata)
# ---------------------------------------------------------------------------
def kalibrer():
    """Regner relativiteter per polise og kalibrerer grunnprisen på treningsdata."""
    for df in (tren, test):
        df["rel_f"] = relativ(df, "frekv")
        df["rel_s"] = relativ(df, "alv")
    b_f = tren["N"].sum() / (tren["Exposure"] * tren["rel_f"]).sum()
    for df in (tren, test):
        df["forv_N"] = b_f * df["Exposure"] * df["rel_f"]
    b_s = tren["kost"].sum() / (tren["forv_N"] * tren["rel_s"]).sum()
    for df in (tren, test):
        df["rp_tariff"] = df["forv_N"] * b_s * df["rel_s"]
    return b_f, b_s


base_f, base_s = kalibrer()

# ---------------------------------------------------------------------------
# 4b. Kredibilitetsjustering av bonus-malus (bare treningsdata)
#     Ny faktor = gammel faktor x (A/E på trening)^Z, der Z = vekten på klassens
#     egne data. Full vekt ved 1 082 forventede skader, det klassiske kravet til
#     full kredibilitet: 90 % sannsynlighet for at observert frekvens ligger
#     innenfor ±5 % av den sanne. Z = kvadratroten av forventede skader / 1 082.
#     Til slutt tvinges faktorene til å være stigende: flere malus-poeng skal
#     aldri gi lavere pris.
# ---------------------------------------------------------------------------
FULL_KRED = 1082
v = "BonusMalus"
F = FAKT[(v, "frekv")]
g = F["grenser"]
n = len(g)
ref = klasse_idx([REF[v]], g)[0]
idx = klasse_idx(tren[v], g)
faktisk = np.bincount(idx, weights=tren["N"].to_numpy(), minlength=n)
forv = np.bincount(idx, weights=tren["forv_N"].to_numpy(), minlength=n)
ae = faktisk / forv
Z = np.minimum(1, np.sqrt(forv / FULL_KRED))
log_just = np.log(F["f"]) + Z * np.log(ae)
log_mono = IsotonicRegression(increasing=True).fit_transform(
    np.arange(n), log_just, sample_weight=forv)
f_ny = np.exp(log_mono - log_mono[ref])

print("\n=== 4b. Kredibilitetsjustering av bonus-malus (frekvensfaktor) ===")
print(pd.DataFrame({"f_gammel": F["f"], "AE_tren": ae, "forv_skader": forv, "Z": Z,
                    "f_justert": np.exp(log_just - log_just[ref]), "f_ny": f_ny},
                   index=pd.Index(etiketter(g), name="klasse")))
F["f"] = f_ny
base_f, base_s = kalibrer()

grunnpris = base_f * base_s
snitt_tren = tren["rp_tariff"].sum() * (1 + PAASLAG) / tren["Exposure"].sum()
print("\n=== 4. Grunnpris (referansekunden, per eksponeringsår) ===")
print(f"Frekvens {base_f:.4f} skader per år x snittskade {base_s:,.0f} euro = {grunnpris:,.0f} euro")
print(f"Med storskadepåslag: {grunnpris * (1 + PAASLAG):,.0f} euro")
print(f"Snittpremie i trening med påslag: {snitt_tren:,.0f} euro per år "
      f"(skal være rundt 173, rå risikopremie i data)")

# ---------------------------------------------------------------------------
# 5. Tarifftabeller med A/E på trening
# ---------------------------------------------------------------------------
def klasse_etikett(df, v):
    if v in KONT:
        g = KONT[v]["grenser"]
        lab = etiketter(g)
        return pd.Categorical(np.array(lab)[klasse_idx(df[v], g)], categories=lab, ordered=True)
    return pd.Categorical(df[v].astype(str), categories=[str(c) for c in KAT_NIVAA[v]])


print("\n=== 5. Tarifftabeller (faktorer relativt til referanseklassen) ===")
print("AE_tren: faktisk/forventet antall skader på trening. stoy_pm: grovt 95 %-bånd")
tabeller = []
for v in [*KONT, *KAT]:
    d = pd.DataFrame({"klasse": klasse_etikett(tren, v), "eksp": tren["Exposure"],
                      "f_frekv": policefaktor(tren, v, "frekv"),
                      "f_alv": policefaktor(tren, v, "alv"),
                      "N": tren["N"], "forv": tren["forv_N"]})
    t = d.groupby("klasse", observed=True).agg(
        eksp=("eksp", "sum"), f_frekv=("f_frekv", "mean"), f_alv=("f_alv", "mean"),
        N=("N", "sum"), forv=("forv", "sum"))
    t["f_samlet"] = t["f_frekv"] * t["f_alv"]
    t["AE_tren"] = t["N"] / t["forv"]
    t["stoy_pm"] = 2 * np.sqrt(OVERDISP / t["forv"])
    t = t[["eksp", "f_frekv", "f_alv", "f_samlet", "AE_tren", "stoy_pm"]]
    if v in ("VehBrand", "Region"):
        t = t.sort_values("eksp", ascending=False)
    print(f"\n--- {v} ---")
    print(t)
    tabeller.append(t.reset_index().assign(variabel=v))

pd.concat(tabeller).to_csv(OUT / "tariff_faktorer.csv", index=False)
with open(OUT / "tariff_grunnpris.json", "w") as fh:
    json.dump({"base_frekvens": base_f, "base_snittskade": base_s,
               "grunnpris_kappet": grunnpris, "storskadepaaslag": PAASLAG,
               "grunnpris_med_paaslag": grunnpris * (1 + PAASLAG),
               "referanse": "45 år, bil 7 år, BM 50, tetthet 300, største kategori i trening"},
              fh, indent=2)
for navn, df in [("tren", tren), ("test", test)]:
    df[["IDpol", "forv_N", "rp_tariff"]].to_parquet(OUT / f"pred_tariff_{navn}.parquet")

# ---------------------------------------------------------------------------
# 6. Test: hvor mye kostet klasseinndelingen?
# ---------------------------------------------------------------------------
def gini(faktisk, forv, eksp):
    rekkef = np.argsort(forv / eksp, kind="mergesort")
    x = np.r_[0, np.cumsum(eksp[rekkef])] / eksp.sum()
    y = np.r_[0, np.cumsum(faktisk[rekkef])] / faktisk.sum()
    return 1 - 2 * np.trapezoid(y, x)


print("\n=== 6. Test: tariff mot full GAM ===")
e_test = test["Exposure"].to_numpy()
for navn, fakt, tar, gam in [("frekvens", "N", "forv_N", "mu_gam"),
                             ("kostnad", "kost", "rp_tariff", "rp_gam")]:
    y = test[fakt].to_numpy()
    print(f"{navn}: A/E tariff {y.sum() / test[tar].sum():.3f}, GAM {y.sum() / test[gam].sum():.3f} | "
          f"Gini tariff {gini(y, test[tar].to_numpy(), e_test):.4f}, "
          f"GAM {gini(y, test[gam].to_numpy(), e_test):.4f}")

forhold = test["rp_tariff"] / test["rp_gam"]
pct = np.percentile(forhold, [1, 5, 25, 50, 75, 95, 99])
print("Tariffpris / GAM-pris per polise, persentiler 1, 5, 25, 50, 75, 95, 99:")
print("  " + "  ".join(f"{p:.3f}" for p in pct))

# ---------------------------------------------------------------------------
# 7. Plott: GAM-kurve mot tariffklasser
# ---------------------------------------------------------------------------
GRID = {"DrivAge": np.arange(18, 91), "VehAge": np.arange(0, 21),
        "BonusMalus": np.arange(50, 151),
        "Density": np.unique(np.geomspace(1, 27000, 300).round())}
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, (v, cfg) in zip(axes.flat, KONT.items()):
    grid = GRID[v]
    Xg = np.tile(X_tren[0], (len(grid), 1))
    Xg[:, cfg["kol"]] = np.log(grid) if v == "Density" else grid
    Ff, Fs = FAKT[(v, "frekv")], FAKT[(v, "alv")]
    kurve = np.exp(np.ravel(gam_f.partial_dependence(term=cfg["kol"], X=klipp_flat(Xg))) - Ff["logref"]
                   + np.ravel(gam_s.partial_dependence(term=cfg["kol"], X=klipp_sev(klipp_flat(Xg))))
                   - Fs["logref"])
    trapp = Ff["f"][klasse_idx(grid, Ff["grenser"])] * Fs["f"][klasse_idx(grid, Fs["grenser"])]
    ax.plot(grid, kurve, color="tab:orange", label="GAM-kurve (frekvens x alvorlighet)")
    ax.step(grid, trapp, where="post", color="tab:blue", label="tariffklasser")
    ax.axhline(1, color="black", lw=0.6)
    if v == "Density":
        ax.set_xscale("log")
    ax.set_title(v), ax.set_ylabel("faktor (referanseklassen = 1)"), ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "tariff_kurver.png", dpi=120)
print(f"\nLagret: tariff_faktorer.csv, tariff_grunnpris.json, pred_tariff_*.parquet, tariff_kurver.png")