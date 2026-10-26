# 01_utforsk.py: A) datakvalitet på alle data, B) testsplit, C) enveis-tabeller på trening
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent    # prosjektmappen (mappen over code/)
OUT = ROOT / "results"

pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_rows", 100)
freq = pd.read_parquet(OUT / "freq_raa.parquet")
sev = pd.read_parquet(OUT / "sev_raa.parquet")

# ---------- A. Datakvalitet (alle data: vi leter etter feil, ikke sammenhenger) ----------
print("Manglende verdier freq:", freq.isna().sum().sum(), "| sev:", sev.isna().sum().sum())
print("Dupliserte IDpol:", freq["IDpol"].duplicated().sum())
print(freq.describe().T[["min", "50%", "max"]])
print("\nEksponering > 1:", (freq["Exposure"] > 1).sum(), "| <= 0:", (freq["Exposure"] <= 0).sum())
print("\nClaimNb:\n", freq["ClaimNb"].value_counts().sort_index())

print("\nSkadebeløp:\n", sev["ClaimAmount"].describe().round(0))
print("Beløp <= 0:", (sev["ClaimAmount"] <= 0).sum())
print("Persentiler:\n", sev["ClaimAmount"].quantile([.5, .9, .99, .995, .999]).round(0))

# Stemmer ClaimNb i polisefilen med antall rader i skadefilen?
print("\nSkaderader uten polise:", (~sev["IDpol"].isin(freq["IDpol"])).sum())
n_sev = sev.groupby("IDpol").size().rename("n_sev")
k = freq.set_index("IDpol")[["ClaimNb"]].join(n_sev).fillna({"n_sev": 0})
k["status"] = np.select(
    [k.ClaimNb == k.n_sev, (k.ClaimNb > 0) & (k.n_sev == 0), k.ClaimNb > k.n_sev],
    ["stemmer", "ClaimNb>0 men ingen beløp", "færre beløp enn ClaimNb"],
    default="flere beløp enn ClaimNb")
print(k["status"].value_counts())

# ---------- B. Sett av 20 % testdata NÅ, før vi ser på sammenhenger ----------
# Alt vi lærer om sammenhenger skal komme fra treningsdata, ellers lekker
# testdata inn i modellvalgene. Fast random_state gjør at splitten blir lik ved
# hver kjøring.
tren_id, test_id = train_test_split(freq["IDpol"], test_size=0.2, random_state=1)
pd.DataFrame({"IDpol": tren_id}).to_parquet(OUT / "tren_id.parquet")
pd.DataFrame({"IDpol": test_id}).to_parquet(OUT / "test_id.parquet")
tren = freq[freq["IDpol"].isin(tren_id)].copy()
print(f"\nTrening: {len(tren):,} poliser | Test: {len(test_id):,}")

# ---------- C. Enveis-tabeller, kun treningsdata ----------
belop = sev.groupby("IDpol")["ClaimAmount"].agg(belop="sum", n_belop="size")
tren = tren.join(belop, on="IDpol").fillna({"belop": 0, "n_belop": 0})

grenser = {  # right=False: [18, 21) betyr 18–20 år
    "DrivAge":    [18, 21, 25, 30, 40, 50, 60, 70, 80, 120],
    "VehAge":     [0, 1, 2, 5, 10, 15, 20, 120],
    "BonusMalus": [50, 51, 60, 70, 80, 90, 100, 120, 150, 231],  # 50 = beste bonus
    "Density":    [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 30000],
}
for kol, g in grenser.items():
    tren[kol + "_g"] = pd.cut(tren[kol], g, right=False)

def enveis(df, kol):
    g = df.groupby(kol, observed=True).agg(
        poliser=("IDpol", "size"), eksponering=("Exposure", "sum"),
        skader=("ClaimNb", "sum"), belop=("belop", "sum"), n_belop=("n_belop", "sum"))
    g["eksp_%"] = 100 * g["eksponering"] / g["eksponering"].sum()
    g["frekvens"] = g["skader"] / g["eksponering"]     # skader per eksponeringsår
    g["snittbelop"] = g["belop"] / g["n_belop"]        # per skade med beløp
    g["risikopremie"] = g["belop"] / g["eksponering"]  # euro per eksponeringsår
    return g[["poliser", "eksp_%", "skader", "frekvens", "snittbelop", "risikopremie"]].round(
        {"eksp_%": 1, "frekvens": 4, "snittbelop": 0, "risikopremie": 0})

# Frekvensen her bruker rå ClaimNb fra polisefilen. I 02_rens.py telles bare
# skader som har et beløp i skadefilen, kappet på 4 per polise, og det gir en
# lavere frekvens (0,0738 mot 0,1010 i trening).
print(f"\nTotal frekvens: {tren.ClaimNb.sum() / tren.Exposure.sum():.4f} | "
      f"snittbeløp: {tren.belop.sum() / tren.n_belop.sum():,.0f}")
for kol in ["DrivAge_g", "VehAge_g", "BonusMalus_g", "Density_g",
            "VehPower", "Area", "VehGas", "VehBrand", "Region"]:
    print(f"\n=== {kol} ===\n{enveis(tren, kol)}")

# ---------- D. Ekstra sjekker ----------
# 1) Har nye biler kortere eksponering? Er VehAge 0-hoppet konsentrert i B12?
tren["ny_bil"] = tren["VehAge"] == 0
print("\nSnitteksponering, ny bil vs. eldre:\n", tren.groupby("ny_bil")["Exposure"].mean().round(3))
print(enveis(tren, ["VehBrand", "ny_bil"]).loc[["B1", "B2", "B12"]])

# 2) Hvor ligger de største skadene, og hvor stor andel av totalen står de for?
print("\nStørste skader i trening:\n",
      tren.nlargest(5, "belop")[["DrivAge", "VehAge", "BonusMalus", "VehPower", "Region", "belop"]])
alle = sev["ClaimAmount"]
print(f"Andel av total skadekostnad fra de 10 største skadene: {alle.nlargest(10).sum() / alle.sum():.1%}")

# 3) Finnes det standardbeløp?
print("\nVanligste beløp:\n", alle.round(2).value_counts().head(5))