
import os
import time
import warnings
import ee
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .gee_utils import (init, rect, land_mask, embeddings, prep_landsat, prep_s2,
                        clearest_landsat, fc_to_df)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

S2B = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI", "NDBI", "MNDWI"]
LS_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

MODELS = {
    "Linear": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))),
    "RF": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                        random_state=C.SEED),
    "GB": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                random_state=C.SEED),
    "FCN": lambda: make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=C.SEED)),
}


# ---------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------
def raw_features(region, year):
    """Conventional inputs (no GFM). Thermal bands are deliberately excluded."""
    cs_plus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
          .linkCollection(cs_plus, ["cs_cdf"]).map(prep_s2)
          .map(lambda i: i.addBands([i.normalizedDifference(["B11", "B8"]).rename("NDBI"),
                                     i.normalizedDifference(["B3", "B11"]).rename("MNDWI")]))
          .select(S2B))
    s2_pct = s2.reduce(ee.Reducer.percentile([10, 50, 90]))           # e.g. B2_p10, B2_p50, B2_p90
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region)
          .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .select(["VV", "VH"])
          .reduce(ee.Reducer.median().combine(ee.Reducer.stdDev(), sharedInputs=True)))
    ls = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
          .filterBounds(region).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
          .filter(ee.Filter.lt("CLOUD_COVER", 30)).map(prep_landsat)
          .select(LS_BANDS).median())
    dem = ee.Image("USGS/SRTMGL1_003").rename("ELEV")
    return ee.Image.cat([s2_pct, s1, ls, dem])


def feature_sets(raw_names):
    raw_median = [f"{b}_p50" for b in S2B] + ["VV_median", "VH_median"] + LS_BANDS + ["ELEV"]
    return {
        "Annual median (no GFM)": raw_median,
        "Multi-temporal (no GFM)": raw_names,
        "AlphaEarth embeddings": C.EMB_BANDS,
        "Embeddings + multi-temporal": C.EMB_BANDS + raw_names,
    }


# ---------------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------------
def sample_cities():
    frames, raw_names = [], None
    for city, bbox in C.CITIES.items():
        region = rect(bbox)
        lst = (prep_landsat(clearest_landsat(C.YEAR, region)).select("LST_C").rename("LST"))
        raw = raw_features(region, C.YEAR)
        if raw_names is None:
            raw_names = raw.bandNames().getInfo()
        stack = (lst.updateMask(land_mask()).addBands(raw)
                 .addBands(embeddings(C.YEAR, region)).addBands(ee.Image.pixelLonLat()).clip(region))
        fc = stack.sample(region=region, scale=30, numPixels=C.N_PER_CITY * 2, seed=C.SEED,
                          geometries=False, tileScale=8).limit(C.N_PER_CITY)
        d = fc_to_df(fc); d["city"] = city; frames.append(d)
        print(f"{city:10s}: {len(d)} samples")
    data = pd.concat(frames, ignore_index=True).dropna(subset=["LST"] + raw_names + C.EMB_BANDS)
    data["y"] = data["LST"] - data.groupby("city")["LST"].transform("mean")    # Eq. 2
    data["block"] = (np.floor(data["longitude"] / C.BLOCK_DEG).astype(int) * 100000 +
                     np.floor(data["latitude"] / C.BLOCK_DEG).astype(int))
    return data, raw_names


# ---------------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------------
def table1_spatial_cv(dub, fsets):
    rows = []
    for fs, feats in fsets.items():
        for m, make in MODELS.items():
            for k, (tr, te) in enumerate(GroupKFold(C.N_FOLDS).split(dub, groups=dub["block"])):
                p = make().fit(dub.loc[tr, feats], dub.loc[tr, "y"]).predict(dub.loc[te, feats])
                rows.append({"input": fs, "model": m, "fold": k,
                             "R2": r2_score(dub.loc[te, "y"], p),
                             "RMSE": np.sqrt(mean_squared_error(dub.loc[te, "y"], p))})
        print("Table I:", fs, "done")
    return pd.DataFrame(rows)


def table2_label_efficiency(dub, fsets):
    rng = np.random.default_rng(C.SEED)
    blocks = dub["block"].unique()
    test_blocks = rng.choice(blocks, size=int(C.LABEL_TEST_FRACTION * len(blocks)), replace=False)
    test, pool = dub[dub["block"].isin(test_blocks)], dub[~dub["block"].isin(test_blocks)]
    rows = []
    for fs in ["Multi-temporal (no GFM)", "AlphaEarth embeddings"]:
        feats = fsets[fs]
        for m, make in MODELS.items():
            for n in [s for s in C.LABEL_SIZES if s <= len(pool)]:
                for r in range(C.LABEL_REPEATS):
                    tr = pool.sample(n, random_state=C.SEED + r)
                    p = make().fit(tr[feats], tr["y"]).predict(test[feats])
                    rows.append({"input": "GFM" if "embeddings" in fs else "No GFM",
                                 "model": m, "n": n, "R2": r2_score(test["y"], p)})
        print("Table II:", fs, "done")
    return pd.DataFrame(rows)


def table3_transfer(data, fsets):
    z = lambda s: (s - s.mean()) / s.std()
    dub = data[data["city"] == "Dubai"]
    rows = []
    for fs in ["Multi-temporal (no GFM)", "AlphaEarth embeddings"]:
        feats = fsets[fs]
        for m, make in MODELS.items():
            mdl = make().fit(dub[feats], z(dub["LST"]))
            for city in [c for c in C.CITIES if c != "Dubai"]:
                te = data[data["city"] == city]
                rows.append({"input": "GFM" if "embeddings" in fs else "No GFM", "model": m,
                             "city": city, "r": pearsonr(z(te["LST"]), mdl.predict(te[feats]))[0]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------
def main():
    init()
    t0 = time.time()
    data, raw_names = sample_cities()
    data.to_csv(os.path.join(C.RESULTS_DIR, "benchmark_samples.csv"), index=False)
    fsets = feature_sets(raw_names)
    dub = data[data["city"] == "Dubai"].reset_index(drop=True)

    t1 = table1_spatial_cv(dub, fsets)
    t1.to_csv(os.path.join(C.RESULTS_DIR, "table1_folds.csv"), index=False)
    tab1 = t1.groupby(["input", "model"])["R2"].mean().unstack()[list(MODELS)].round(3)
    tab1.to_csv(os.path.join(C.RESULTS_DIR, "table1_spatial_cv_R2.csv"))
    print("\nTable I — spatial CV R²\n", tab1.to_string())

    t2 = table2_label_efficiency(dub, fsets)
    t2.to_csv(os.path.join(C.RESULTS_DIR, "table2_runs.csv"), index=False)
    tab2 = t2.groupby(["model", "input", "n"])["R2"].mean().unstack("n").round(3)
    tab2.to_csv(os.path.join(C.RESULTS_DIR, "table2_label_efficiency.csv"))
    print("\nTable II — label efficiency\n", tab2.to_string())

    t3 = table3_transfer(data, fsets)
    tab3 = t3.pivot_table(index="model", columns=["city", "input"], values="r").round(3)
    tab3.to_csv(os.path.join(C.RESULTS_DIR, "table3_transfer_r.csv"))
    print("\nTable III — cross-city transfer r\n", tab3.to_string())
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. Results in '{C.RESULTS_DIR}/'.")


if __name__ == "__main__":
    main()
