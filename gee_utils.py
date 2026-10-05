"""Earth Engine helpers shared by all experiments."""
import ee
import pandas as pd
from .config import PROJECT_ID, EMB_BANDS


# ---------------------------------------------------------------------------------
# Initialisation & basic layers
# ---------------------------------------------------------------------------------
def init(project: str = PROJECT_ID):
    """Initialise Earth Engine (authenticates on first use)."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def rect(bbox):
    return ee.Geometry.Rectangle(bbox)


def worldcover():
    """ESA WorldCover 2021 v200 (class 50 = built-up, 60 = bare, 80 = water)."""
    return ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")


def land_mask():
    return worldcover().neq(80)


def embeddings(year: int, region):
    """AlphaEarth Foundations annual 64-D embeddings (10 m)."""
    return (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filterBounds(region).mosaic().select(EMB_BANDS))


# ---------------------------------------------------------------------------------
# Landsat & Sentinel-2 preprocessing
# ---------------------------------------------------------------------------------
def prep_landsat(img):
    """Cloud-mask a Landsat C2 L2 image; return scaled SR bands, LST (°C) and NDVI."""
    qa = img.select("QA_PIXEL")
    clear = (qa.bitwiseAnd(1 << 1).eq(0)             # dilated cloud
             .And(qa.bitwiseAnd(1 << 3).eq(0))        # cloud
             .And(qa.bitwiseAnd(1 << 4).eq(0)))       # cloud shadow
    optical = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    lst = (img.select("ST_B10").multiply(0.00341802).add(149.0)
           .subtract(273.15).rename("LST_C"))
    out = optical.addBands(lst).updateMask(clear)
    ndvi = out.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")
    return ee.Image(out.addBands(ndvi).copyProperties(img, ["system:time_start"]))


def prep_s2(img):
    """Sentinel-2 SR masked with Cloud Score+ (cs_cdf >= 0.6), scaled to reflectance."""
    clear = img.select("cs_cdf").gte(0.60)
    refl = img.select("B.*").divide(10000)
    ndvi = refl.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return ee.Image(refl.addBands(ndvi).updateMask(clear)
                    .copyProperties(img, ["system:time_start"]))


def clearest_landsat(year, region, months=(6, 9), sensors=("LC08", "LC09"), max_cloud=10):
    """Least-cloudy Landsat C2 L2 scene in the given months (raw image)."""
    col = None
    for s in sensors:
        c = ee.ImageCollection(f"LANDSAT/{s}/C02/T1_L2")
        col = c if col is None else col.merge(c)
    end_month = months[1] + 1
    end = f"{year}-{end_month:02d}-01" if end_month <= 12 else f"{year + 1}-01-01"
    col = (col.filterBounds(region).filterDate(f"{year}-{months[0]:02d}-01", end)
           .filter(ee.Filter.lt("CLOUD_COVER", max_cloud)).sort("CLOUD_COVER"))
    return ee.Image(col.first())


def landsat_lst(year, region, **kw):
    """LST (°C) of the clearest summer scene."""
    return prep_landsat(clearest_landsat(year, region, **kw)).select("LST_C").rename("LST")


# ---------------------------------------------------------------------------------
# Anomalies, hotspots, areas
# ---------------------------------------------------------------------------------
def anomaly(img, region, smooth_m=60):
    """Within-scene anomaly (Eq. 2): smoothed value minus land mean, water masked."""
    land = land_mask()
    v = (img.rename("v").focalMean(radius=smooth_m, units="meters")
         .updateMask(img.mask()).updateMask(land))
    m = v.reduceRegion(ee.Reducer.mean(), region, 30, maxPixels=1e10, tileScale=4).get("v")
    return v.subtract(ee.Number(m)).rename("v")


def pct_threshold(img, region, pct):
    d = img.reduceRegion(ee.Reducer.percentile([pct], ["p"]), region, 30,
                         maxPixels=1e10, tileScale=4)
    return ee.Number(ee.Dictionary(d).values().get(0))


def hot_mask(img, region, pct=90):
    """Eq. 3: H_i = 1 if anomaly exceeds the P-th percentile."""
    return img.gt(pct_threshold(img, region, pct)).rename("hot")


def area_km2(mask01, region):
    a = (ee.Image.pixelArea().divide(1e6).updateMask(mask01.selfMask())
         .reduceRegion(ee.Reducer.sum(), region, 30, maxPixels=1e10, tileScale=4)
         .get("area"))
    return ee.Number(ee.Algorithms.If(a, a, 0))


# ---------------------------------------------------------------------------------
# FeatureCollection -> pandas
# ---------------------------------------------------------------------------------
def fc_to_df(fc) -> pd.DataFrame:
    """Download a FeatureCollection as a DataFrame (properties only)."""
    try:
        df = ee.data.computeFeatures({"expression": fc, "fileFormat": "PANDAS_DATAFRAME"})
        return df.drop(columns=[c for c in ["geo"] if c in df.columns])
    except Exception:
        feats = fc.getInfo()["features"]
        return pd.DataFrame([f["properties"] for f in feats])
