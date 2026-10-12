# LarvaLens

Computer vision portfolio repo on the public black soldier fly (BSF) larvae dataset: Zenodo 10.5281/zenodo.13359376 (CC BY 4.0), linked paper Nawoya et al. 2025, Smart Agricultural Technology 11, 100953. Goal: a polished public GitHub repo that shows clean Python engineering, careful data processing and publication-quality figures. Not a paper.

Owner: Md Raihan Patwary, ORCID 0009-0006-0616-3742. GitHub: raihan-patwary. Kaggle: mdraihanpatwary02.

Task plan: @PLAN.md. Work through it in order and tick each box when it is done.

## Where work happens
- Local (this folder): code, tests, figures, README, git.
- Kaggle through the kaggle CLI: the 6.9 GB photo zip and all GPU work. One folder per notebook, kernels/NN_name/, with kernel-metadata.json (is_private true; enable_gpu and enable_internet only when needed). Chain notebooks with kernel_sources so each download happens once.
- Run a notebook: kaggle kernels push -p kernels/NN_name, poll kaggle kernels status, then kaggle kernels output <user/slug> -p outputs/NN_name.
- Repo files a notebook needs (the larvalens package, splits) reach Kaggle as one small private dataset, larvalens-support, updated with kaggle datasets version and attached with dataset_sources.
- Never commit data, zips, photos, model weights or the paper PDF (CC BY-NC-ND). Small CSVs go to results/, final figures to figures/.

## Data files (Zenodo record 13359376)
| File | Size | md5 |
| --- | --- | --- |
| BSF_Larvae_v1.zip | 13.3 MB | a5a104e2c10bbc499ad3c45b71d2ffd8 |
| Larva_sex_classification.zip | 226.3 MB | 0e28870daeb6bb4ffb1142aeb4811206 |
| Larvae--size-data.xlsx | 98.1 kB | 8aa2c6926db0d1554c07a99d12dd060d |
| Single Larvae data.zip | 6.9 GB | ea931d09ac8533f909d73371d45bf43d |

URL pattern: https://zenodo.org/records/13359376/files/<URL-encoded file name>?download=1

## Expected values: verify, never force
Check each one and write results/data_checks.md (expected, found, match). If a value differs, report and explain it. Never edit data or this list to make a value match.
- Detection set: 321 images, over 5,000 larva instances
- Single-larva photos: 1,608
- Larva Data sheet: 1,613 rows
- Weight (g): complete, 0.058 to 0.353
- Length (mm): complete, 11 to 26
- Width (mm): 92% filled, 52 zeros (errors, treat as missing)
- Sex: 91% filled, 1,079 female, 383 male
- Batch_ID: 86% filled, 219 blank, 12 named batches; B011 and B013 have no notes
- Batch_Info: hatch dates for 10 batches, temperature (25 or 29 °C) for 9
- Rows with both age and temperature: 1,001
- Grouped_Larva sheet: 24 rows of group weights (keep out of models)
- Pearson r, Weight vs Length: 0.86

## Paper benchmarks (README comparison only)
Checked against the paper PDF (Nawoya et al. 2025). Section numbers refer to the paper.
- Detection: segmentation (mask) AP50 0.93 at IoU 0.5 (3.1). YOLOv8m-seg, COCO-pretrained, 100 epochs, 640 x 640 input; 224 train and 97 test images (2.4.1, Table 1). Our v8n and v8s models are smaller than theirs; say so beside the number.
- Weight from shape: R2 0.80, RMSE 0.018 g, linear regression on area, length and width (Table 2, 3.2). Area and length alone gave R2 0.77. Split 80/20, 1,280 train and 320 test larvae (2.5.1, Table 1). Table 2 lists RMSE 0.25 for "area and width", likely a typo for 0.025; quote only the 0.80 row.
- Weight from image: R2 0.71, RMSE 0.023 g (3.2, Fig. 6). ResNet101 ImageNet features plus SVR. Crops were resized to 224 x 224 and the original crop size was fed back into the regression (2.5.3), so size information was not lost.
- Sex: image-only ResNet18, accuracy 74%, F1 0.75, test n = 139 (3.3). Plain accuracy; the paper does not say which class the F1 refers to. The data was cut down to 309 male and 386 female to balance it (2.6), so their 74% is on near-balanced data and is not directly comparable to our balanced accuracy on the full imbalanced set. Fusion model (ResNet18 plus MLP on weight and length): 69%, test n = 136.
- The paper says 80/20 train/test but never says how the split was drawn. Call our comparison split "a random 80/20 split, seed 42", not "the paper's split".

## Paper facts about the setup (for checks and the data card)
- Imaging box with a Raspberry Pi HQ camera (12.3 MP) and an 8 to 50 mm zoom lens, 70 mm above the larva, diffuse LED light, white background (2.2, 4.1).
- Single-larva photos are 4056 x 3040 px (2.4.2).
- Calibration: 28 px per mm, from a reference object in the images (2.5.1). Compare our fitted mm per pixel with this and report the difference; never force it.
- Length by digital caliper, weight by microscale (1 mg precision). The method for width is not stated. Sex was recorded when each larva emerged as an adult, from a labelled vial (2.3).
- Larvae were 11 to 14 days after hatching, 16 batches, June to October 2023 (2.1). The file has 12 named batches, July to November; report the file's numbers.
- The paper does not describe how photo names link to larva IDs. Find this from the files.

## Engineering standards
- Python 3.11, src layout (src/larvalens), pyproject.toml with pinned versions.
- Type hints and docstrings on every public function; small, pure functions.
- One CLI: python -m larvalens <command>. Configs in configs/*.yaml. Seed 42 everywhere.
- pytest tests on synthetic data only, so CI never downloads anything. ruff for lint and format.
- GitHub Actions runs ruff and pytest on every push. Keep it green.
- Notebooks tell the story; logic lives in src/.
- Splits are saved once in splits/ and reused by every model. See "Splits" below.

## Splits (decided 2026-10-12)
- Main evaluation: 5-fold cross-validation grouped by batch, seed 42. A batch never sits in both train and test. Report the mean and spread (SD) across folds.
- Balance folds on rearing temperature (25 °C, 29 °C, unknown) wherever the batches allow it.
- B011 and B013 go into the folds like any other batch. Image and shape models never use age or temperature, so their missing notes do not matter.
- The 219 larvae with no batch are left out of cross-validation, because they might come from the same rearing round as a test batch. They are a bonus test set for the final model (trained on all batched rows), reported separately.
- Figures use each larva's out-of-fold prediction (from the fold where it was held out).
- Anything fitted on data (mm per pixel, scalers, hyperparameters) is fitted inside each fold on its training rows only.
- Paper comparison: one extra random 80/20 split of all larvae, seed 42, reported only beside the paper's numbers.
- Sex model: the same batch folds if the crops link back to spreadsheet rows. If not, 5 folds stratified by sex, seed 42, and the README says so.
- splits/larvae.csv holds one row per larva with its fold (or "no_batch") and its random-split role.

## Image and model rules
- Weight from pixels: the same fixed crop window for every larva; never resize a crop to fill the frame; no zoom or scale augmentation. Flips and 90 degree rotations are fine; keep colour jitter light.
- mm per pixel is fitted on training rows only.
- Sex model: class weights; report balanced accuracy, per-class F1 and ROC-AUC.

## Figures (src/larvalens/viz, style file bsf.mplstyle)
- Okabe-Ito colours with fixed meaning: female #D55E00, male #0072B2, 25 °C #56B4E9, 29 °C #E69F00. viridis for continuous values.
- Widths 3.5 in or 7.2 in. Export SVG and 300 dpi PNG. No top or right spines. Direct labels over legends. Panel letters. README figures get a one-line takeaway title.
- fig01 hero banner, 1280 x 640 px, light and dark versions
- fig02 mosaic of 60 crops, lightest to heaviest
- fig03 missing-value matrix by batch, plus photo-to-row match count
- fig04 batch timeline, hatch to measurement days, coloured by temperature
- fig05 weight ridgeline per batch, 25 °C vs 29 °C
- fig06 image QC, blur vs brightness, outliers as thumbnails
- fig07 detection overlay, ground truth beside prediction
- fig08 predicted vs true count per image, 1:1 line
- fig09 training curves, loss and mask mAP per epoch
- fig10 morphometrics explainer on one larva
- fig11 calibration (pixel vs mm length) and Bland-Altman
- fig12 allometry, log weight vs log length with power-law fit
- fig13 predicted vs measured weight, one panel per model
- fig14 model comparison dot plot, paper values as reference marks
- fig15 failure gallery, worst weight predictions
- fig16 sex confusion matrix, ROC and PR curves
- fig17 Grad-CAM gallery, right and wrong, male and female
- fig18 UMAP of DINOv2 features as thumbnails, plus coloured by weight
- fig19 demo GIF (stretch)

## Writing rules
- Never use em dashes anywhere: README, docstrings, comments, commit messages. Use commas, parentheses or separate sentences.
- Plain, short sentences.
- Credit the dataset (Geissmann 2024, Zenodo, CC BY 4.0) and cite Nawoya et al. 2025 in the README, DATA_CARD.md and figure captions.
- Repo license AGPL-3.0, because Ultralytics is AGPL-3.0.

## Workflow
- Work through PLAN.md block by block without waiting for me between blocks.
- End every block with: tests pass, README updated, one clear commit, push, and a short summary that lists the new figure paths.
- If a Kaggle run fails twice, stop and tell me. Report expected-value mismatches in results/data_checks.md and keep going.
- Ask me before: making the repo or any Kaggle notebook public, creating a Hugging Face Space, deleting files, force-pushing, tagging a release.
