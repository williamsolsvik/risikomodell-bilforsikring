# 03_baseline_glm.py: Poisson-GLM med grupperte variabler = "dagens tariff"
from pathlib import Path
import json
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
pd.set_option("display.width", 200, "display.max_rows", 200)

tren = pd.read_parquet(OUT / "tren.parquet")
test = pd.read_parquet(OUT / "test.parquet")

GRENSER = {  # right=False: [18, 21) betyr 18–20 år
    "DrivAge":    [18, 21, 25, 30, 40, 50, 60, 70, 80, 120],
    "VehAge":     [0, 1, 2, 5, 10, 15, 20, 120],
    "BonusMalus": [50, 51, 60, 70, 80, 90, 100, 120, 231],   # 120+ slått sammen (tynt)
    "Density":    [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 30000],
}
KATEGORISKE = ["VehBrand", "VehGas", "Region"]

def lag_grupper(df):
    df = df.copy()
    for kol, g in GRENSER.items():
        df[kol + "_g"] = pd.cut(df[kol], g, right=False).astype(str)
    for kol in KATEGORISKE:
        df[kol] = df[kol].astype(str)
    return df

tren, test = lag_grupper(tren), lag_grupper(test)

# Referansenivå = gruppen med mest eksponering i trening (får faktor 1,00)
variabler = [k + "_g" for k in GRENSER] + KATEGORISKE
ref = {v: tren.groupby(v)["Exposure"].sum().idxmax() for v in variabler}
formel = "N ~ " + " + ".join(f'C({v}, Treatment(reference="{ref[v]}"))' for v in variabler)

# Poisson med log-link og offset log(eksponering)
glm = smf.glm(formel, data=tren, family=sm.families.Poisson(),
              offset=np.log(tren["Exposure"])).fit()

# Faktortabell: exp(koeffisient) = multiplikativ faktor relativt til referansegruppen
ki = glm.conf_int()
faktorer = pd.DataFrame({"faktor": np.exp(glm.params), "nedre_95": np.exp(ki[0]),
                         "ovre_95": np.exp(ki[1]), "p": glm.pvalues}).round(3)
faktorer.index = (faktorer.index
                  .str.replace(r'C\((\w+), Treatment\(reference=".*?"\)\)\[T\.', r"\1 = ", regex=True)
                  .str.rstrip("]"))
print("Referansegrupper:", ref)
print(f"Grunnfrekvens for referansekunden: exp(Intercept) = {np.exp(glm.params['Intercept']):.4f}\n")
print(faktorer)

# Overdispersjon: Pearson chi2 / frihetsgrader. Nær 1 = Poisson passer
overdisp = glm.pearson_chi2 / glm.df_resid
print(f"\nOverdispersjon: {overdisp:.3f}")

# Deviance på trening og test, mot en nullmodell (samme frekvens for alle)
tren["mu_glm"] = glm.fittedvalues                                   # forventet antall skader
test["mu_glm"] = glm.predict(test, offset=np.log(test["Exposure"]))
pois = sm.families.Poisson()
frek_snitt = tren.N.sum() / tren.Exposure.sum()
for navn, d in [("Trening", tren), ("Test", test)]:
    dev_null = pois.deviance(d.N, d.Exposure * frek_snitt) / len(d)
    dev_glm = pois.deviance(d.N, d.mu_glm) / len(d)
    print(f"{navn}: deviance per polise | nullmodell {dev_null:.5f} | GLM {dev_glm:.5f} | "
          f"forbedring {1 - dev_glm / dev_null:.2%}")
print(f"Faktisk / forventet (A/E) totalt på test: {test.N.sum() / test.mu_glm.sum():.3f}")

# Lagre prediksjoner og faktorer til steg 6–8
tren[["IDpol", "mu_glm"]].to_parquet(OUT / "pred_glm_tren.parquet")
test[["IDpol", "mu_glm"]].to_parquet(OUT / "pred_glm_test.parquet")
faktorer.to_csv(OUT / "faktorer_glm.csv")

# Overdispersjonen brukes i støybåndene i steg 6 til 8. Den avrundede verdien er
# den som brukes, den eksakte lagres bare som dokumentasjon. Filen deles med
# steg 5, så vi oppdaterer bare våre egne nøkler.
EST = OUT / "estimater.json"
est = json.loads(EST.read_text()) if EST.exists() else {}
est.update(overdisp=round(overdisp, 2), overdisp_eksakt=overdisp)
EST.write_text(json.dumps(est, indent=2))