# Data checks

Expected values from CLAUDE.md, checked against the downloaded files. Mismatches are reported and explained; the data is never edited to match.

| Item | Expected | Found | Match | Note |
| --- | --- | --- | --- | --- |
| Detection set | 321 images, over 5,000 larva instances | 321 images, 6,612 instances | yes | Only 185 distinct source photos (see findings). |
| Single-larva photos | 1,608 | not yet run | pending |  |
| Larva Data rows | 1,613 | 1,613 | yes |  |
| Weight (g) | complete, 0.058 to 0.353 | 1,613/1,613, 0.058 to 0.353 | yes |  |
| Length (mm) | complete, 11 to 26 | 1,613/1,613, 11.00 to 26.43 | yes | The maximum is 26.43 mm; the expected range was rounded. |
| Width (mm) | 92% filled, 52 zeros | 92% filled (1,491), 52 zeros | yes | Zeros are set to missing in labels.csv. |
| Sex | 91% filled, 1,079 female, 383 male | 91% filled, 1,079 female, 383 male | yes |  |
| Batch_ID | 86% filled, 219 blank, 12 named batches; B011 and B013 have no notes | 86% filled, 219 blank, 12 named; no notes: B011, B013 | yes |  |
| Batch_Info | hatch dates for 10 batches, temperature (25 or 29 °C) for 9 | hatch dates 10, temperature 9 (25, 29 °C) | yes |  |
| Rows with both age and temperature | 1,001 | 1,001 | yes |  |
| Grouped_Larva rows | 24 | 24 | yes | Kept out of models. |
| Pearson r, Weight vs Length | 0.86 | 0.86 | yes |  |

## Other findings

- Detection set: 321 images come from only 185 source photos. It is a Roboflow export with up to 5 augmented copies per photo, so the image split is grouped by source photo to stop copies leaking between train and test.
- Detection labels use two class IDs ({0: 4713, 1: 1899}), but they mark the same thing: in 25 source photos, copies of one photo use class 0 in one copy and class 1 in another. Both are merged into one class, larva.
- Dense detection images are not exhaustively labelled: larvae at the edges and in clumps are often unlabelled, so predicted counts above the labelled count are expected.
- Sex crops: 1,577 files, not sorted into male and female folders. Each is named by larva ID, so sex, batch and weight come from the spreadsheet and the sex model can use the batch folds.
- 36 sex crop files are not images (they hold browser log text) and are excluded.
- Larva IDs 152 and 172 each have two different crops (0152.jpg and 152.jpg; 0172.jpg and 00172.jpg); both IDs are excluded from the sex model.
- Usable sex crops: 1,413 (1,045 female, 368 male), 99 of them with no batch.
- Age at measurement ranges from 10 to 20 days after hatching for the batches with a hatch date. The paper says 11 to 14 days.
- The 114 unbatched larvae measured on 2023-11-18 share that date with batch B013, and the other 105 were measured on 2023-08-27, just after B011. This supports keeping unbatched larvae out of cross-validation.
