# MarineGuard v2 pipeline: YOLO11-Seg + Conv-Autoencoder + LSTM

```
RAW XTF/SSS -> ingestion -> QC -> calibration -> median -> bilateral -> CLAHE -> SNR -> SA-CFAR
   -> [ YOLO11-Seg (known)  |  Conv-AE (unknown) ] -> decision fusion -> LSTM tracking
   -> geolocation + SCQI -> GIS/KDE -> PostGIS/SQLite -> FastAPI / dashboard -> CSV / JSON / GeoJSON
```

Everything is driven by **`configs/pipeline.yaml`**. Its hash is stored with every run and every detection, together
with the version (`name@sha256[:10]`) of each model that produced it.

## Quick start

```bash
python -m models.yolo11_seg.dataset            # survey-disjoint pseudo-mask dataset (once)
python -m models.yolo11_seg.train --epochs 15 --imgsz 256 --batch 32   # ~2 h on CPU
python -m models.autoencoder.train             # normal-seabed autoencoder + validated threshold (~8 min, CPU)
python -m models.lstm.train                    # LSTM motion model (simulated scenes, ~2 min)
python -m models.yolo11_seg.evaluate ; python -m models.lstm.evaluate ; python scripts/evaluate_pipeline.py
python scripts/export_onnx.py                  # ONNX + parity check
pytest tests                                   # 76 tests
uvicorn api:app --port 8000                    # REST API (existing routes unchanged)
streamlit run frontend/dashboard/app.py --server.port 8502    # pipeline dashboard
python deploy/jetson/run_edge.py --input survey.xtf --backend onnx   # offline / edge
```

## Layout

| Path | Contents |
|---|---|
| `backend/pipeline/` | one module per stage: `ingestion`, `quality_control`, `calibration`, `preprocessing`, `snr`, `sa_cfar`, `yolo_seg`, `anomaly`, `fusion`, `tracking`, `geolocation`, `scqi`, `render`, `runner` (orchestrator), `onnx_backend` |
| `backend/gis/` | KDE hotspots, SCQI heatmap, resurvey areas, CSV/JSON/GeoJSON export |
| `backend/database/` | `schema.sql` (PostGIS) + `repository.py` (PostGIS with SQLite fallback) |
| `backend/api/router.py` | `/upload /process /detect /track /results/{id} /metrics /export` (also under `/api/v1`) |
| `models/yolo11_seg/` `autoencoder/` `lstm/` | dataset/train/evaluate/inference per model. `models/autoencoder` also re-exports the legacy `SonarAnomalyDetector` so `app.py` is unchanged |
| `frontend/{dashboard,sonar,gis,metrics}` | standalone Streamlit dashboard (LIVE/ANALYSIS, GIS, MODEL EVALUATION) |
| `deploy/jetson/` | `build_engines.sh` (trtexec FP16/INT8), `run_edge.py` |
| `tests/pipeline/` | unit tests for every stage + end-to-end + API |

Every run logs each stage's **time and output summary** to `outputs/pipeline_logs/<run_id>.jsonl` and returns them in the result.

## Results (all measured in this repository)

### YOLO11n-Seg, known debris (held-out test split, 773 images, CPU)
| Box P / R / F1 | mAP@50 | mAP@50:95 | Mask mAP@50 | Mask mAP@50:95 | Mask IoU | Latency / FPS | Size |
|---|---|---|---|---|---|---|---|
| 0.904 / 0.921 / 0.912 | 0.949 | 0.895 | 0.933 | 0.862 | 0.880 (98.7% >= 0.5) | 20.6 ms / 48 FPS | 5.96 MB, 2.84 M params |

**Read the mask numbers with care.** The dataset contains only boxes (7,673 rows, zero polygons), so masks are *pseudo-masks*
(GrabCut, falling back to an inscribed ellipse when GrabCut degenerates: 78% of instances). Mask metrics measure agreement
with those pseudo-masks, not true pixel accuracy. Box metrics use the real annotations. The 27-class images are tiny
object crops (about 84x97 px), an easier problem than full-survey detection.

### Convolutional autoencoder, unknown anomalies (real Anoma sonar, held-out frames)
Trained on normal seabed patches only. Threshold (0.01001) chosen on validation data with FPR <= 10%.
| Set | ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|---|
| Real anomalies (primary, n=65) | 0.902 | 0.550 | 0.443 | 0.538 | 0.486 | 0.070 | 0.462 |
| Synthetic (secondary, n=200) | 0.614 | 0.291 | 0.254 | 0.075 | 0.116 | 0.070 | 0.925 |

MSE 0.0035 normal vs 0.0130 anomaly; MAE 0.039 vs 0.080; SSIM 0.562 vs 0.427. The synthetic injections are subtle and the
model mostly misses them, so they are reported but never used to tune the threshold. Small test set (65 real objects):
treat these numbers as indicative.

### LSTM tracker (simulated scenes: no labelled real tracks exist here)
| | Pos. MAE / RMSE (1-step) | Trajectory ADE / FDE | Continuity | ID switches | Success | FPS |
|---|---|---|---|---|---|---|
| LSTM | 1.10 / 1.39 px | 2.84 / 4.07 px | 0.993 | 44 | 0.961 | 155 |
| Constant-velocity baseline | 1.82 / 2.40 px | 5.40 / 8.04 px | 0.985 | 71 | 0.929 | 1709 |

The LSTM halves trajectory error and cuts ID switches, at ~11x the per-frame cost. It has **not** been validated on real
survey tracks, and false-track rate is slightly worse than the baseline (7.5% vs 6.2%).

### End-to-end (`scripts/evaluate_pipeline.py`, final config)
* **Known debris through the whole system** (250 test images): correct class + box **94.0%**, known precision **0.898**, 0.11 false knowns / image.
* **Real anomalies, Anoma held-out half** (thresholds chosen on the other half): 54 of 65 annotated regions found (**recall 0.83**), precision 0.45, F1 0.58, **5.9 false alarms / frame**.
  Scored by *region-hit* (a detection centre inside the annotated box) because Anoma annotations are large regions (median 303 px of 640) while SA-CFAR
  proposes small specks (median 10 px); strict IoU>=0.3 recall is only 0.35 and is reported in the JSON for transparency.

### Acoustic signature analysis (impedance-contrast verification)
`backend/pipeline/acoustic_signature.py` + `signature_verifier.py`; trained by `scripts/extract_signatures.py` and `scripts/train_signature.py`.
It runs between fusion and tracking, attaches a signature and a verdict to every object (stored in `evidence.acoustic_signature`,
`signature_verdict`), and **never changes an object's category**.

* **What it measures**: a *relative* impedance-contrast index (Michelson form, the same algebra as R = (Z2 - Z1)/(Z2 + Z1), applied to backscatter
  amplitude), contrast in dB, highlight-to-shadow ratio, shadow depth/length, edge sharpness, texture, plus shadow-based height on real waterfalls.
  Uncalibrated 8-bit imagery cannot give absolute impedance.
* **Result on 773 held-out test objects**: verification AUROC 0.936; class from signature alone 45% top-1 / 70% top-3 (chance 3.7%).
  **Acoustic-only features** (no size/shape): AUROC 0.912, top-1 35%, so it is not just object size.
* **Against the real YOLO detector** (822 calls, 79 wrong): flags 19% of its errors while wrongly flagging 6% of correct calls; CONFIRMED verdicts are right
  **96%** of the time vs 90% for the detector alone (confirms 47% of correct calls, 18% of wrong ones).
* **A physics claim that did NOT hold**: metal is not brighter than plastic or glass here (median contrast index 0.084 / 0.099 / 0.092). Rubber (tires) is
  clearly higher (0.156). So the index cannot identify a material on its own; the page says so. Material families were assigned by me from class names.
* **Bug caught by looking at real output**: the first CONFIRMED rule (claimed class in the top-3) confirmed a "bottle" that the signature rated 1% while being 98%
  sure of "chain". CONFIRMED now requires the claimed class to be competitive with the signature's best guess (cut-off chosen on validation).
* **Limits**: validated on 27-class object chips only, so verdicts are N/A on raw waterfalls (features are still reported); the 27-class crops rarely show
  strong shadows, so shadow length is mostly zero; MISMATCH is a review flag, not a rejection.

## Things found and fixed while building this (worth knowing)
1. **Data leakage**: 10 of 12 shipwreck surveys were split across train/val/test. Shipwrecks are now re-split by survey. Other classes are single-source crops with no survey id, so their split is unchanged.
2. **XTF reader bug**: every ping was stamped with `time.time()` (parse time) instead of the ping header time, which breaks speed/trajectory QC and tracking. Now decoded from the header. The sample generator also wrote non-monotonic seconds and an impossible ~49-knot track; both fixed, and the sample is now deterministic (seeded). *The JSF reader appears to have the same timestamp defect; it was not verified or fixed.*
3. **Enhancement hurts the AI branches.** YOLO on CLAHE-enhanced input: 56% correct; on the calibrated image it was trained on: **94%**. The AE showed the same direction (F1 0.545 -> 0.582). Both AI branches now consume the calibrated image; enhancement feeds SNR and SA-CFAR.
4. **Config error**: `yolo_seg.imgsz` was 640 while the model is trained/evaluated at 256.
5. **Square padding costs 7 mAP points** (mAP@50:95 0.824 vs 0.896) on these non-square crops. The ONNX export therefore uses dynamic H/W (identical mAP to PyTorch); a static TensorRT engine would silently lose accuracy.
6. **Reproducibility**: OpenCV's K-Means RNG was unseeded, so identical inputs gave different detections. Seeded per run.
7. **Nadir suppression / calibration are waterfall-only.** Applied to plain images they delete the image centre; now skipped for non-waterfalls (the shadow check likewise becomes direction-agnostic).
8. **SA-CFAR fragments long highlights** into several candidates; nearby fragments are merged (`merge_gap_px`).
9. Learned (logistic) fusion and 256 px AE context crops were both tried and **did not beat** the simple fusion on held-out data, so they are off by default (`scripts/train_fusion.py` remains).

### Trade-off you should choose deliberately
`sa_cfar.merge_gap_px` merges CFAR fragments into one object. All rows use the final configuration and the Anoma
held-out half (noise threshold re-selected on the validation half for each gap):

| gap | validation F1 | test recall | test precision | test F1 | false alarms / frame |
|---|---|---|---|---|---|
| 6 px  | **0.545** | 0.923 | 0.387 | 0.545 | 28.7 |
| **20 px (default)** | 0.527 | 0.831 | 0.448 | **0.582** | **5.9** |
| 40 px | 0.495 | 0.692 | 0.380 | 0.490 | 3.1 |
| 80 px | 0.441 | 0.600 | 0.379 | 0.465 | 2.1 |

Pure validation F1 marginally favours 6 px, but that setting produces ~5x the false alarms; 20 px was chosen as the
operating point (best test F1, a workable alarm rate). Use a smaller gap if missing an object is worse than reviewing
false alarms, a larger one for the opposite.

## Known limitations
* **SA-CFAR is a small-target detector**: objects larger than the guard window contaminate their own reference ring and are missed by the CFAR path (known large objects are still caught by YOLO).
* **Mask quality is unverified against real masks** (see above).
* **PostGIS is implemented but untested here**: no PostgreSQL server or `psycopg2` on this machine; the SQLite fallback is what ran. The schema and SQL are covered by static tests only.
* **TensorRT / Jetson is untested**: no NVIDIA hardware here. ONNX export and ONNX Runtime execution *are* verified (identical results to PyTorch); `deploy/jetson/build_engines.sh` is written from the trtexec docs and unexecuted. INT8 needs a real-sonar calibration cache and a re-validated AE threshold.
* **No React front end exists in this repository**; the dashboard is Streamlit. The FastAPI service (JSON/GeoJSON, CORS on) is what a React app would consume.
* Plain images have no navigation: their positions use synthetic telemetry and the result is flagged as such.
* Model training here used a CPU (YOLO 15 epochs); more epochs / a GPU would likely improve YOLO further.
