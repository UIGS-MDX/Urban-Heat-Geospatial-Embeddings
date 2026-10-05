"""
Multi-year hotspot mapping (paper Section III-C, Table IV, Fig. 2).

  * RF regressor trained on 2025 AlphaEarth embeddings -> 2025 Landsat LST anomaly
  * Applied to 2017–2024 embeddings and validated against independent Landsat scenes
  * Hotspots = hottest 10 % of land (Eq. 3); persistent / emerging / disappeared hotspots
  * Repeated within built-up land (WorldCover class 50) for intra-urban hotspots
  * Exports the GeoTIFFs used to draw Fig. 2

Run:  python -m src.hotspots            (add --no-export to skip Drive exports)
"""
import os
import sys
import time
import ee
import numpy as np
import pandas as pd

from . import config as C
from .gee_utils import (init, rect, worldcover, land_mask, embeddings, landsat_lst,
                        anomaly, hot_mask, area_km2, fc_to_df)

WC_NAMES = {10: "Trees", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
            60: "Bare/sand", 80: "Water", 90: "Wetland", 95: "Mangroves"}


def f1_iou(obs_hot, pred_hot, region):
    c = (ee.Image.cat([obs_hot.And(pred_hot), obs_hot, pred_hot]).rename(["tp", "o", "p"])
         .reduceRegion(ee.Reducer.sum(), region, 30, maxPixels=1e10, tileScale=4).getInfo())
    tp, o, p = c["tp"], c["o"], c["p"]
    return (2 * tp / (o + p) if o + p else np.nan,
            tp / (o + p - tp) if o + p - tp else np.nan)


def pattern_r(pred, obs, region, scale):
    r = (pred.rename("p").addBands(obs.rename("o"))
         .reduceRegion(ee.Reducer.pearsonsCorrelation(), region, scale,
                       maxPixels=1e10, tileScale=4).getInfo())
    return r.get("correlation")


def reverse_geocode(lat, lon):
    """Place name via OpenStreetMap Nominatim (optional dependency: geopy)."""
    try:
        from geopy.geocoders import Nominatim
        loc = Nominatim(user_agent="dubai_uhi_gfm").reverse((lat, lon), zoom=14, language="en")
        a = loc.raw.get("address", {}) if loc else {}
        time.sleep(1)                                    # Nominatim usage policy
        place = (a.get("suburb") or a.get("neighbourhood") or a.get("village") or a.get("town")
                 or a.get("city_district") or a.get("city") or "")
        return place, a.get("state", "")
    except Exception:
        return "", ""


def main(export=True):
    init()
    roi = rect(C.ROI_BBOX)
    wc, land = worldcover(), land_mask()
    built = wc.eq(50)

    # ---------------- train on the 2025 scene ----------------
    obs_anom = {}
    for y in C.YEARS:
        try:
            obs_anom[y] = anomaly(landsat_lst(y, roi), roi, C.SMOOTH_M)
        except Exception as e:
            print(f"{y}: no observed scene ({e})")
    train_fc = (embeddings(C.TRAIN_YEAR, roi).addBands(obs_anom[C.TRAIN_YEAR].rename("target"))
                .sample(region=roi, scale=30, numPixels=8000, seed=C.SEED, tileScale=4))
    rf = (ee.Classifier.smileRandomForest(numberOfTrees=200, minLeafPopulation=3, seed=C.SEED)
          .setOutputMode("REGRESSION").train(train_fc, "target", C.EMB_BANDS))

    pred_anom = {y: anomaly(embeddings(y, roi).classify(rf), roi, C.SMOOTH_M) for y in C.YEARS}

    # ---------------- Table IV: temporal validation ----------------
    rows = []
    for y in C.YEARS:
        row = {"year": y}
        if y in obs_anom:
            o, p = obs_anom[y], pred_anom[y].updateMask(obs_anom[y].mask())
            row["r_city"] = pattern_r(p, o, roi, 60)
            row["F1_city"], row["IoU_city"] = f1_iou(hot_mask(o, roi, C.HOT_PCT),
                                                      hot_mask(p, roi, C.HOT_PCT), roi)
            ou, pu = o.updateMask(built), p.updateMask(built)
            row["r_builtup"] = pattern_r(pu, ou, roi, 30)
            row["F1_builtup"], row["IoU_builtup"] = f1_iou(hot_mask(ou, roi, C.HOT_PCT),
                                                            hot_mask(pu, roi, C.HOT_PCT), roi)
        rows.append(row); print(y, "validated")
    t4 = pd.DataFrame(rows).set_index("year").round(3)
    unseen = t4.drop(index=C.TRAIN_YEAR, errors="ignore")
    t4.loc["Mean unseen"] = unseen.mean().round(3)
    t4.to_csv(os.path.join(C.RESULTS_DIR, "table4_temporal_validation.csv"))
    print("\nTable IV\n", t4.to_string())

    # ---------------- persistence / emerging / disappeared ----------------
    pred_hot = {y: hot_mask(pred_anom[y], roi, C.HOT_PCT) for y in C.YEARS}
    pred_hot_u = {y: hot_mask(pred_anom[y].updateMask(built), roi, C.HOT_PCT) for y in C.YEARS}
    persistence = (ee.ImageCollection([pred_hot[y].unmask(0).toInt() for y in C.YEARS])
                   .sum().updateMask(land).rename("years_hot"))
    persist_u = (ee.ImageCollection([pred_hot_u[y].unmask(0).toInt() for y in C.YEARS])
                 .sum().updateMask(built).rename("years_hot"))
    early = ee.ImageCollection([pred_hot[y].unmask(0) for y in C.YEARS[:3]]).max()
    late = ee.ImageCollection([pred_hot[y].unmask(0) for y in C.YEARS[-3:]]).max()
    classes = {"Persistent": persistence.gte(C.PERSIST_MIN),
               "Emerging": late.And(early.Not()),
               "Disappeared": early.And(late.Not()),
               "Persistent intra-urban": persist_u.gte(C.PERSIST_MIN)}
    dyn = pd.Series({k: area_km2(v, roi).getInfo() for k, v in classes.items()}, name="area_km2").round(1)
    dyn.to_csv(os.path.join(C.RESULTS_DIR, "hotspot_dynamics_km2.csv"))
    print("\nHotspot dynamics (km²)\n", dyn.to_string())

    comp = {}
    for k in ["Persistent", "Emerging", "Disappeared"]:
        g = (ee.Image.pixelArea().divide(1e6).updateMask(classes[k].selfMask()).addBands(wc)
             .reduceRegion(ee.Reducer.sum().group(1, "c"), roi, 30, maxPixels=1e10, tileScale=4)
             .get("groups").getInfo())
        d = pd.DataFrame(g); d["class"] = d["c"].map(WC_NAMES)
        comp[k] = (100 * d.set_index("class")["sum"] / d["sum"].sum()).round(1)
    comp = pd.DataFrame(comp).fillna(0)
    comp.to_csv(os.path.join(C.RESULTS_DIR, "hotspot_landcover_percent.csv"))
    print("\nLand cover of hotspot types (%)\n", comp.to_string())

    # ---------------- top persistent intra-urban patches ----------------
    vec = (persist_u.gte(C.PERSIST_MIN).selfMask().addBands(pred_anom[C.TRAIN_YEAR])
           .reduceToVectors(geometry=roi, scale=30, geometryType="polygon", eightConnected=True,
                            labelProperty="label", reducer=ee.Reducer.mean(),
                            maxPixels=1e10, tileScale=4))
    vec = vec.map(lambda f: f.set({"area_ha": f.geometry().area(1).divide(1e4),
                                   "lon": f.geometry().centroid(1).coordinates().get(0),
                                   "lat": f.geometry().centroid(1).coordinates().get(1)}))
    top = (vec.filter(ee.Filter.gte("area_ha", C.MIN_PATCH_HA))
           .sort("mean", False).limit(C.TOP_PATCHES))
    tu = fc_to_df(top).rename(columns={"mean": "anomaly_C"})
    tu = tu[["anomaly_C", "area_ha", "lat", "lon"]].round(4)
    names = [reverse_geocode(r.lat, r.lon) for r in tu.itertuples()]
    tu["place"], tu["emirate"] = [n[0] for n in names], [n[1] for n in names]
    tu.index = range(1, len(tu) + 1)
    tu.to_csv(os.path.join(C.RESULTS_DIR, "top_urban_hotspots.csv"), index_label="rank")
    print("\nTop persistent intra-urban hotspots\n", tu.to_string())

    # ---------------- exports for Fig. 2 ----------------
    if export:
        stack = ee.Image.cat([obs_anom[C.TRAIN_YEAR].rename("anom2025"),
                              persistence.unmask(0).rename("persist"),
                              persist_u.gte(C.PERSIST_MIN).unmask(0).rename("urbanhot"),
                              wc.rename("wc")]).toFloat()
        for img, name in [(stack, "fig2_layers"),
                          (pred_anom[C.TRAIN_YEAR].rename("pred2025").toFloat(), "fig2_pred2025")]:
            ee.batch.Export.image.toDrive(image=img, description=name, folder=C.DRIVE_FOLDER,
                                          region=roi, scale=60, crs="EPSG:4326",
                                          maxPixels=1e10).start()
        print(f"\nExports started to Drive/{C.DRIVE_FOLDER}: fig2_layers.tif, fig2_pred2025.tif")
        print(f"Copy them into '{C.DATA_DIR}/' and run: python -m src.figures")


if __name__ == "__main__":
    main(export="--no-export" not in sys.argv)
