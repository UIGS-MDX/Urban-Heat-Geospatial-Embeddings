# Mapping Surface Urban Heat Patterns Using Geospatial Foundation Model Embeddings

Code accompanying the paper:

> P. G. Prathibha, M. S. Seena, K. Nannath, S. Kaitheri, S. Balasubramanian,
> *Mapping Surface Urban Heat Patterns Using Geospatial Foundation Model Embeddings*,
> Urban Intelligence and Geosensing Lab, Middlesex University Dubai.

We evaluate Google's **AlphaEarth Foundations** embeddings (64 bands, 10 m, annual) for
detecting and characterizing surface urban heat islands (SUHIs) in hyper-arid Dubai, and
compare them with conventional Sentinel-1, Sentinel-2 and Landsat inputs. All processing
runs in **Google Earth Engine + Python on CPU** — no GPU or model fine-tuning is required.

## Repository structure

```
dubai-uhi-gfm/
├── src/
│   ├── config.py         # study areas, years, seeds and all experiment parameters
│   ├── gee_utils.py      # Earth Engine helpers: cloud masking, LST, embeddings, hotspots
│   ├── benchmark.py      # Tables I–III: GFM vs conventional inputs, label efficiency, transfer
│   ├── hotspots.py       # Table IV + Fig. 2 layers: multi-year hotspot mapping 2017–2025
│   ├── lcz.py            # LCZ classification (2018 -> 2025) and SUHI per LCZ
│   ├── shap_drivers.py   # RF + TreeSHAP driver attribution
│   └── figures.py        # Fig. 2 (four-panel hotspot maps)
├── notebooks/
│   └── run_all_colab.ipynb   # runs everything in Google Colab
├── data/                 # place exported GeoTIFFs here (not tracked)
├── results/              # CSV tables and figures are written here (not tracked)
├── requirements.txt
└── LICENSE
```

## What each script reproduces

| Script | Paper output |
|---|---|
| `src.benchmark` | **Table I** (spatial-CV R², 4 inputs × 4 models), **Table II** (label efficiency), **Table III** (cross-city transfer) |
| `src.hotspots` | **Table IV** (temporal validation 2017–2025), hotspot areas (persistent / emerging / disappeared), intra-urban hotspots, exports for Fig. 2 |
| `src.lcz` | LCZ classification accuracy and SUHI intensity per LCZ relative to bare sand |
| `src.shap_drivers` | SHAP ranking of interpretable LST drivers |
| `src.figures` | **Fig. 2** |

## Setup

1. Register a Google Cloud project for Earth Engine: <https://code.earthengine.google.com/register>
2. Install dependencies: `pip install -r requirements.txt`
3. Set your project ID, either in `src/config.py` (`PROJECT_ID`) or as an environment variable:
   ```bash
   export EE_PROJECT=your-gcp-project-id
   ```
4. Authenticate once: `earthengine authenticate` (or run `ee.Authenticate()` in Colab).

## Running

Run from the repository root:

```bash
python -m src.benchmark        # ~15–30 min (samples 4 cities, trains 4 model families)
python -m src.hotspots         # ~10–20 min, starts two Drive exports for Fig. 2
python -m src.lcz              # ~5–10 min
python -m src.shap_drivers     # ~5 min
```

When the exports `fig2_layers.tif` and `fig2_pred2025.tif` have finished
(<https://code.earthengine.google.com/tasks>), copy them from Google Drive (`Dubai_UHI/`)
into `data/`, then:

```bash
python -m src.figures
```

In Google Colab, open `notebooks/run_all_colab.ipynb`, which performs all steps.

## Key settings (see `src/config.py`)

| Parameter | Value |
|---|---|
| Study area (Dubai) | 54.90–55.65 °E, 24.75–25.40 °N |
| Transfer cities | Abu Dhabi, Doha, Riyadh |
| Benchmark year | 2025 (clearest June–September Landsat 8/9 scene) |
| Target | within-scene LST anomaly (LST minus land mean) |
| Spatial CV | ~2 km blocks, 5-fold grouped CV |
| Models | ridge linear probe, RF (200 trees), histogram GB, FCN (128-64) |
| Hotspots | hottest 10 % of land (P90); persistent = ≥ 7 of 9 years |
| Random seed | 42 |

Earth Engine sampling uses fixed seeds, so results are reproducible up to minor
differences from data reprocessing in the Earth Engine catalog.

## Data sources

| Dataset | Earth Engine ID |
|---|---|
| Landsat 8/9 Collection 2 Level-2 | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` |
| Sentinel-2 SR (harmonized) + Cloud Score+ | `COPERNICUS/S2_SR_HARMONIZED`, `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` |
| Sentinel-1 GRD | `COPERNICUS/S1_GRD` |
| AlphaEarth Foundations embeddings | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| ESA WorldCover 2021 | `ESA/WorldCover/v200` |
| Global LCZ map | `RUB/RUBCLIM/LCZ/global_lcz_map/latest` |
| GHSL built-up, height, population | `JRC/GHSL/P2023A/...` |
| VIIRS night-time lights | `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` |
| SRTM elevation | `USGS/SRTMGL1_003` |

The AlphaEarth Foundations Satellite Embedding dataset is released under CC-BY 4.0;
please follow the attribution requirements on its Earth Engine catalog page.

## Citation

If you use this code, please cite the paper (BibTeX will be added after publication).

## Acknowledgement

This work was supported in part by the Dubai Future Foundation under a Research,
Development and Innovation (RDI) Grant.

## License

Code: MIT (see `LICENSE`). Data are subject to the licenses of their respective providers.
