# Data card

## Source

Geissmann, Q. (2024). *Black soldier fly larvae dataset* (version 1). Zenodo. https://doi.org/10.5281/zenodo.13359376. License: CC BY 4.0.

The data backs Nawoya, S., Geissmann, Q., Karstoft, H., Bjerge, K., Akol, R., Katumba, A., Mwikirize, C. and Gebreyesus, G. (2025). Prediction of black soldier fly larval sex and morphological traits using computer vision and deep learning. *Smart Agricultural Technology* 11, 100953. https://doi.org/10.1016/j.atech.2025.100953. The article is CC BY-NC-ND, so no figure from it is reused here.

Nothing from the dataset is stored in this repository. `python -m larvalens data` downloads the files and checks their md5 sums.

## Files

| File | Size | md5 | Used for |
| --- | --- | --- | --- |
| Larvae--size-data.xlsx | 98.1 kB | 8aa2c6926db0d1554c07a99d12dd060d | labels for every task |
| BSF_Larvae_v1.zip | 13.3 MB | a5a104e2c10bbc499ad3c45b71d2ffd8 | detection and segmentation |
| Larva_sex_classification.zip | 226.3 MB | 0e28870daeb6bb4ffb1142aeb4811206 | sex classification |
| Single Larvae data.zip | 6.9 GB | ea931d09ac8533f909d73371d45bf43d | size and weight from photos (processed on Kaggle) |

## How the larvae were measured (from the paper)

- Black soldier fly larvae from a colony at Aarhus University, fed chicken feed, reared at 25 or 29 °C.
- Each larva got an ID, was weighed on a microscale (1 mg precision) and measured for length with a digital caliper, then photographed one at a time.
- Photos came from an imaging box: Raspberry Pi HQ camera (12.3 MP) with an 8 to 50 mm zoom lens, 70 mm above the larva, diffuse LED light, white background. Single-larva photos are 4056 x 3040 px. The paper uses 28 px per mm.
- Each larva was then kept in a labelled vial until it emerged as an adult, when its sex was recorded. Sex labels are therefore adult sex, not a guess from the larva.
- The paper does not say how width was measured.

## Spreadsheet: Larvae--size-data.xlsx

| Sheet | Contents |
| --- | --- |
| Larva Data | 1,613 larvae: Batch_ID, Date, L_ID, Weight (g), Length (mm), Width (mm), Sex |
| Batch_Info | free-text notes on hatch date and rearing temperature for B001 to B010 |
| Grouped_Larva | 24 rows of larvae weighed in groups; not used in any model |
| MetaData, Tools and process | column descriptions and the tools used |

Cleaning (`src/larvalens/clean.py`):

- 52 widths recorded as 0 are errors and become missing (flag `width_zero`).
- Batch_Info notes are parsed into `batches.csv`: hatch dates for 10 batches and temperatures for 9 (B010 has no temperature; B011 and B013 have no notes).
- Age in days = measurement date minus hatch date, where the hatch date is known (1,080 larvae). Ages run from 10 to 20 days; the paper says 11 to 14.
- 219 larvae have no batch. They were measured on 2023-08-27 (105) and 2023-11-18 (114), so they may share a rearing round with B011 or B013. They are kept out of cross-validation and used as a separate bonus test set.

All expected values from the planning brief were confirmed; see `results/data_checks.md`.

## Detection set: BSF_Larvae_v1.zip

- `Images/` and `Labels/`: 321 images at 640 x 640 px and YOLO segmentation labels (class, then polygon points, normalised). There is no ready train, validation and test split.
- It is a Roboflow export. The 321 images come from only 185 source photos; 70 photos appear 2 to 5 times as augmented copies (file names `<source>_jpg.rf.<hash>.jpg`). The split in `splits/detect.csv` keeps all copies of a photo in the same part (train 130, val 28, test 27 source photos).
- 6,612 polygons with class IDs 0 (4,713) and 1 (1,899). Both mean "larva": copies of the same photo use class 0 in one copy and class 1 in another (25 photos). They are merged into one class.
- Crowded images are only partly labelled (larvae at the edges and in clumps are often unlabelled). Count error is therefore measured against the labelled count, and over-counts are expected.
- 2 label files are empty.

## Sex crops: Larva_sex_classification.zip

- 1,577 JPEG crops cut from the single-larva photos, 298 to 1,483 px wide. They are **not** sorted into male and female folders. Each file is named by larva ID (`0001.jpg`), so sex, batch and weight come from the spreadsheet.
- 36 files are not images (they contain browser log text) and are excluded.
- Larva IDs 152 and 172 each have two different crops (`0152.jpg` and `152.jpg`, `0172.jpg` and `00172.jpg`); both IDs are excluded.
- 124 crops belong to larvae with no recorded sex.
- Usable for the sex model: 1,413 crops (1,045 female, 368 male), listed in `splits/sex_crops.csv`.
- The paper trained on a balanced subset of 309 male and 386 female crops. Here all usable crops are kept and the imbalance is handled with class weights and balanced metrics.

## Single-larva photos: Single Larvae data.zip

Processed on Kaggle by `kernels/01_photos` (download, md5, size, blur, brightness, perceptual hash, photo-to-row match, copies with the long side at most 2048 px). The zip holds one folder, `single_larvae_renamed_IDs/`.

- 1,608 JPEGs, all 4056 x 3040 px, named by larva ID (`0001.jpg`). No EXIF rotation.
- Every photo matches exactly one spreadsheet row. Five rows have no photo: larva IDs 215, 368, 1001, 1282 and 1602.
- No byte-identical duplicates. Blur and brightness are even across the set, and no photo is unreadable.
- Downstream steps use the 2048 x 1536 copies (scale 0.505).
- Scale: image length against caliper length gives 30.2 px per mm on the copies (about 60 px per mm on the full photos), steady from 29 to 32 px per mm across all measuring days. The paper states 28 px per mm. The difference is reported, not forced.
- Many photos show a printed ruler and an ID label beside the larva. The YOLO model also detects digits on the labels, so the measuring step keeps the most confident larva-sized mask.

## Splits

See CLAUDE.md, "Splits". In short: 5-fold cross-validation grouped by batch and balanced on rearing temperature; the 219 unbatched larvae as a bonus test set; one random 80/20 split (seed 42) only for comparison with the paper.
