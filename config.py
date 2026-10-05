"""Central configuration. Edit PROJECT_ID (and paths if needed) before running."""
import os

# ---- Google Earth Engine -------------------------------------------------------
PROJECT_ID = os.environ.get("EE_PROJECT", "your-gcp-project-id")   # Cloud project registered for Earth Engine

# ---- Study areas (lon_min, lat_min, lon_max, lat_max) -------------------------
ROI_BBOX = [54.90, 24.75, 55.65, 25.40]                 # Dubai, ~5,450 km²
CITIES = {
    "Dubai":     [54.90, 24.75, 55.65, 25.40],
    "Abu Dhabi": [54.30, 24.30, 54.65, 24.55],
    "Doha":      [51.40, 25.20, 51.62, 25.40],
    "Riyadh":    [46.55, 24.55, 46.90, 24.85],
}

# ---- General -------------------------------------------------------------------
SEED = 42
YEAR = 2025                                             # benchmark / training year
EMB_BANDS = [f"A{i:02d}" for i in range(64)]            # AlphaEarth embedding bands

# ---- Benchmark (Tables I–III) ----------------------------------------------------
N_PER_CITY = 4000                                       # pixels sampled per city at 30 m
BLOCK_DEG = 0.02                                        # ~2 km spatial blocks
N_FOLDS = 5
LABEL_SIZES = [50, 100, 250, 500, 1000, 2000]
LABEL_REPEATS = 5
LABEL_TEST_FRACTION = 0.25

# ---- Multi-year hotspot mapping (Table IV, Fig. 2) ----------------------------
YEARS = list(range(2017, 2026))
TRAIN_YEAR = 2025
SMOOTH_M = 60                                           # focal-mean radius before thresholding
HOT_PCT = 90                                            # hotspot = hottest 10 % of land
PERSIST_MIN = 7                                         # persistent = hot in >= 7 of 9 years
MIN_PATCH_HA = 5
TOP_PATCHES = 10

# ---- LCZ characterization --------------------------------------------------------
LCZ_TRAIN_YEAR, LCZ_TARGET_YEAR = 2018, 2025
LCZ_MIN_PROB = 60                                       # % ensemble agreement in global LCZ map
LCZ_PER_CLASS = 400
LCZ_MIN_CLASS = 30

# ---- Paths -----------------------------------------------------------------------
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
DATA_DIR = os.environ.get("DATA_DIR", "data")           # local copies of exported GeoTIFFs
DRIVE_FOLDER = "Dubai_UHI"                              # Google Drive folder for GEE exports
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
