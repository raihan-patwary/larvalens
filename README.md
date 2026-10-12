# LarvaLens

From photos to phenotypes: counting, measuring, weighing and sexing black soldier fly larvae with computer vision.

[![CI](https://github.com/raihan-patwary/larvalens/actions/workflows/ci.yml/badge.svg)](https://github.com/raihan-patwary/larvalens/actions/workflows/ci.yml)
[![Dataset DOI](https://img.shields.io/badge/data-10.5281%2Fzenodo.13359376-blue)](https://doi.org/10.5281/zenodo.13359376)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

Work in progress. This README grows block by block.

## Results so far

### Counting and outlining larvae

YOLOv8 instance segmentation on the 321 crowded images. The split keeps every augmented copy of a photo on the same side (train 130, validation 28, test 27 source photos), so test photos are truly unseen.

| Model | Mask AP50 | Mask AP50-95 | Box AP50 | Count error (median, larvae per image) |
| --- | --- | --- | --- | --- |
| YOLOv8n-seg | 0.87 | 0.63 | 0.87 | +2 |
| YOLOv8s-seg | 0.87 | 0.63 | 0.87 | +3 |
| YOLOv8m-seg | **0.89** | **0.66** | 0.88 | +2 |
| Nawoya et al. (2025), YOLOv8m-seg | 0.93 | | | |

Our numbers are a little below the paper's. Two things explain most of the gap. The paper's 224/97 split does not say how augmented copies were handled, so copies of a test photo may have been in training. And many larvae in the crowded photos were never labelled, so correct detections of them count as false positives and as over-counts.

![Labelled and predicted larvae](figures/fig07_detection_overlay.png)

![Predicted vs labelled counts](figures/fig08_count_agreement.png)

![Training curves](figures/fig09_training_curves.png)

### Measuring and weighing larvae from their shape

The best YOLO model (YOLOv8m-seg) outlined the larva in all 1,608 single-larva photos (median confidence 0.96). From each outline we measure area, perimeter, skeleton length and width, and fit mm per pixel inside each fold on training larvae only.

![How a larva is measured](figures/fig10_morphometrics.png)

| Model | R² (batch folds, mean ± SD) | MAE, mg (batch folds) | R² (unbatched bonus set) | R² (random 80/20) | MAE, mg (random 80/20) |
| --- | --- | --- | --- | --- | --- |
| Linear (area, length, width) | 0.53 ± 0.17 | 20.1 | 0.25 | 0.58 | 18.6 |
| Gradient boosting | 0.51 ± 0.12 | 20.4 | 0.26 | 0.60 | 17.7 |
| Power law (length) | 0.42 ± 0.18 | 22.5 | 0.02 | 0.44 | 21.2 |
| Nawoya et al. (2025), linear, random 80/20 | | | | 0.80 | RMSE 18 mg |

Our shape models explain less of the weight than the paper reports, even on a random split (R² 0.58 to 0.60 against 0.80). We checked the obvious suspects. The camera scale is steady across all 25 measuring days (29 to 32 px per mm on the 2048 px copies). The limit is the agreement between image length and caliper length (r = 0.73, ±2.7 mm): larvae stretch and contract between the caliper and the camera, which the paper also mentions. Testing on new batches costs a further 0.02 to 0.09 in R², and the unbatched larvae are harder still because their weights span a narrow range.

We also measured 30 px per mm on the 2048 px copies, which is about 60 px per mm on the full photos. The paper states 28 px per mm. We report both and do not force a match.

![Image length against caliper length](figures/fig11_calibration.png)

![Predicted against measured weight](figures/fig13_pred_vs_measured.png)

![Model comparison](figures/fig14_model_comparison.png)

![Allometry](figures/fig12_allometry.png)

![Crops from lightest to heaviest](figures/fig02_mosaic.png)

![Image quality](figures/fig06_image_qc.png)

### The data

All expected values in the planning brief were confirmed (`results/data_checks.md`). The audit also found problems worth knowing about, listed in [DATA_CARD.md](DATA_CARD.md).

![Data audit](figures/fig03_data_audit.png)

![Batch timeline](figures/fig04_batch_timeline.png)

![Weight by batch](figures/fig05_weight_ridgeline.png)

## Quickstart

```bash
uv sync                      # Python 3.11 environment with pinned packages
uv run python -m larvalens data      # download, verify, clean, audit and split
uv run python -m larvalens figures   # rebuild every figure from saved results
uv run pytest
```

GPU steps run as Kaggle notebooks in `kernels/`.

## Data

Geissmann, Q. (2024). Black soldier fly larvae dataset. Zenodo. https://doi.org/10.5281/zenodo.13359376 (CC BY 4.0).

Nawoya, S., et al. (2025). Prediction of black soldier fly larval sex and morphological traits using computer vision and deep learning. Smart Agricultural Technology 11, 100953. https://doi.org/10.1016/j.atech.2025.100953
