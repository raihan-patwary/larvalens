# PLAN

Work through the blocks in order and tick each box when it is done. Follow CLAUDE.md throughout. No need to wait for me between blocks; only stop where CLAUDE.md says to ask.

## Before block 0
- [x] Check `gh auth status` and `kaggle kernels list --mine`. If either fails, stop and tell me how to fix it.

## Block 0: setup
- [x] git init here; create a private GitHub repo named larvalens with gh; push.
- [x] src/larvalens package (src layout), pyproject.toml with pinned versions, Makefile (data, test, figures), .gitignore that excludes data/, outputs/, *.zip, *.pt and *.pth.
- [x] src/larvalens/viz/bsf.mplstyle plus a small viz helper that applies the figure rules.
- [x] ruff and pytest with one smoke test; a GitHub Actions workflow that runs both.
- [x] Private Kaggle dataset larvalens-support containing the package.
- [x] Commit, push, CI green.

## Block 1: data and cross-check
- [x] Download Larvae--size-data.xlsx, BSF_Larvae_v1.zip and Larva_sex_classification.zip into data/raw and verify each md5. If a file is already in data/raw, only verify it.
- [x] src/larvalens/clean.py: load the Larva Data sheet, set zero widths to missing, parse Batch_Info into data/processed/batches.csv (batch, hatch date, temperature), add age in days, save data/processed/labels.csv.
- [x] Run every expected-value check in CLAUDE.md and write results/data_checks.md.
- [x] Unzip both zips. Record the annotation format, folder layout, image sizes, file naming, and whether the sex crops carry IDs that link to spreadsheet rows. Start DATA_CARD.md with this.
- [x] Save splits/larvae.csv as CLAUDE.md describes.
- [x] kernels/01_photos (internet on, no GPU): download Single Larvae data.zip, check md5, record size, blur (variance of Laplacian) and brightness per photo, match photo names to spreadsheet rows, and save qc.csv, match.csv and copies with the long side at most 2048 px to its output. Push it and keep working while it runs. When pulling its outputs, skip the photo copies.
- [x] Unit tests for clean.py on small synthetic tables.
- [x] Figures 3, 4 and 5.
- [x] Commit, push.

## Block 2: detection and segmentation
- [x] kernels/02_detect (GPU, internet on): download BSF_Larvae_v1.zip, check md5, convert labels to YOLO segmentation format if needed, make a fixed image split with seed 42 (pull it into splits/detect.csv), and train YOLOv8n-seg, YOLOv8s-seg and YOLOv8m-seg (the paper's size, for a like-for-like comparison). Save to its output: test-set box and mask precision, recall, mAP50 and mAP50-95; predicted vs true count per test image; loss and mAP per epoch; best weights; 6 test overlays.
- [x] Pull outputs to outputs/02_detect. Metrics go to results/; weights stay out of git.
- [x] Figures 7, 8 and 9 in the house style, not the Ultralytics default plots.
- [x] README results table with our numbers beside the paper's AP50 0.93.
- [x] Commit, push.

## Block 3: morphometrics and weight from shape
- [x] src/larvalens/morphometrics.py: from a binary mask compute area, perimeter, skeleton length (longest path, using skan), width (twice the max of the distance transform, and mean width over the middle half of the skeleton) and mean Lab colour. Unit tests on synthetic shapes with known answers: a straight capsule and a bent one.
- [x] kernels/03_measure (GPU, kernel_sources 01_photos and 02_detect): segment every photo with the best YOLO model (fall back to an Otsu threshold when no mask is found, and flag those rows), run morphometrics, and save measurements.csv, fixed-size crops centred on each larva, and 12 example panels.
- [x] Locally: in each batch fold, fit mm per pixel on training rows only (image length vs measured Length) and report agreement on held-out rows; compare with the paper's 28 px per mm. Fit weight models (linear on area, length and width; power law W = a L^b; gradient boosting) with the batch folds in splits/larvae.csv, then the no-batch bonus test set, plus the random 80/20 split for comparison with the paper.
- [x] Figures 2, 6, 10, 11 and 12, plus first versions of 13 and 14.
- [x] Commit, push.

## Block 4: weight from pixels and embeddings
- [ ] kernels/04_pixels (GPU, internet on, kernel_sources 03_measure): fine-tune a timm ResNet-18 on the fixed crops to predict weight (Huber loss on log weight; flips and 90 degree rotations only) with the batch folds in splits/larvae.csv (one model per fold, then the no-batch bonus test set and the random split). Extract DINOv2 ViT-S/14 features for every crop, fit ridge regression on them, and compute a 2D UMAP. Save predictions, metrics, features and UMAP coordinates.
- [ ] Figures 13 (all models), 14 (final), 15 and 18.
- [ ] Commit, push.

## Block 5: sex classification and Grad-CAM
- [ ] kernels/05_sex (GPU, internet on): download Larva_sex_classification.zip, check md5, use the batch folds if the crops link to spreadsheet rows (else 5 folds stratified by sex with seed 42, and say so in the README), and fine-tune a timm ResNet-18 with class weights. Save accuracy, balanced accuracy, per-class F1, ROC-AUC, the confusion matrix, ROC and PR curve data, and Grad-CAM overlays for 8 right and 8 wrong crops per class.
- [ ] Figures 16 and 17.
- [ ] Commit, push.

## Block 6: README and v1.0
- [ ] Figure 1, the hero banner (1280 x 640 px), from one test larva: photo, mask, skeleton with width, predicted weight and sex. Light and dark versions.
- [ ] README in this order: title and the pitch "From photos to phenotypes: counting, measuring, weighing and sexing black soldier fly larvae with computer vision."; badges (dataset DOI, license, Python, CI); hero banner in a <picture> tag for light and dark mode; what it does (count, size, weight, sex); results table, ours beside the paper's; quickstart (install, make data, make figures); pipeline diagram in Mermaid; figure gallery; data credit and citation; author line and a one-line note that it was built with Claude Code.
- [ ] CITATION.cff, LICENSE (AGPL-3.0), finished DATA_CARD.md, CI green.
- [ ] Remind me to switch the repo on in Zenodo (GitHub integration) before the release, so v1.0 gets a DOI.
- [ ] Show me the README. After I approve, make the repo public and create release v1.0 with short notes.
- [ ] Tell me how to set the repo's social preview image to the hero banner.

## Stretch: live demo (only when I ask)
- [ ] app/ with Gradio: a visitor uploads a photo; a crowded photo returns masks and a count, and a single larva returns length, width, area, estimated weight and sex probability with its Grad-CAM heatmap. Small models so it runs on CPU. Ask me before creating the Hugging Face Space, then deploy it, put the weights on the Hub with model cards, and add the demo link and badge to the README.
