# Evaluation Report — Run 2026-09-08_v1

Full run of `AI_Video_Detection_Project.ipynb` on the DFDC sample dataset (77 real / 323 fake videos),
using the notebook's actual configured hyperparameters (no epoch shortcuts). Wall-clock time: ~2h 5m.

## Configuration

| Setting | Value |
|---|---|
| NUM_FRAMES | 8 |
| IMG_SIZE | 224 |
| BATCH_SIZE | 4 |
| EPOCHS_BASELINE | 8 |
| EPOCHS_LSTM | 8 (early-stopped) |
| TUNE_EPOCHS | 2 |
| DROPOUT_P | 0.3 |
| VAL_SPLIT / TEST_SPLIT | 0.15 / 0.15, stratified by class |
| SEED | 42 |

## Data split (stratified real/fake)

| Split | Real | Fake | Total |
|---|---|---|---|
| Train | 55 | 227 | 282 |
| Val | 11 | 48 | 59 |
| Test | 11 | 48 | 59 |

## Training summary

**Baseline (CNN, ResNet18):** ran the full 8 epochs. Best validation loss (0.4632, val_acc 0.8136) was
at epoch 7; that checkpoint was restored. Val loss was noisy epoch-to-epoch (0.56 → 0.66 → 0.63 → 0.54 →
0.47 → 0.64 → **0.46** → 0.47), which is expected with only 59 validation videos — a couple of
misclassifications swing val_loss noticeably.

**Hyperparameter tuning (CNN+LSTM, 2 epochs/config):**

| hidden_size | lr | val_loss |
|---|---|---|
| 128 | 1e-4 | 0.4490 |
| **256** | **1e-4** | **0.4251** (best) |
| 256 | 5e-5 | 0.4507 |

**Final CNN+LSTM (hidden_size=256, lr=1e-4):** early-stopped at epoch 4 (patience=3). Best checkpoint was
actually **epoch 1** (val_loss 0.4256); epochs 2–4 each got slightly worse, triggering the stop.

## Test set results (11 real / 48 fake, n=59)

| Metric | CNN Baseline | CNN + LSTM |
|---|---|---|
| Accuracy | 0.7966 | 0.7797 |
| Precision | 0.8214 | 0.8182 |
| Recall | 0.9583 | 0.9375 |
| F1 | 0.8846 | 0.8738 |
| **ROC-AUC** | **0.4943** | **0.4659** |

**Per-class (CNN Baseline):** real — precision 0.33, recall **0.09**, f1 0.14 (support 11); fake — precision 0.82, recall 0.96, f1 0.88 (support 48).

**Per-class (CNN + LSTM):** real — precision 0.25, recall **0.09**, f1 0.13 (support 11); fake — precision 0.82, recall 0.94, f1 0.87 (support 48).

**Error analysis:** CNN+LSTM misclassified 13/59 (22%) test clips — 10 of the 11 real videos were predicted
as fake (recall 0.09 ≈ 1/11 correct), with fairly confident wrong probabilities (`prob_fake` 0.70–0.94, see
`misclassified_list.txt`); the remaining ~3 errors were fake videos predicted as real.

## Interpretation — read this before accuracy alone

The headline accuracy numbers (~78–80%) look passable in isolation, but two things make that misleading:

1. **A trivial "always predict fake" classifier scores ~81% accuracy** on this test split (48/59) — higher
   than either trained model achieved. Accuracy alone is actively hiding what's going on here, which is
   exactly why the notebook's own design rationale (§9) argues for reporting precision/recall/F1/ROC-AUC
   rather than accuracy alone — this run is a direct, concrete illustration of that point.
2. **ROC-AUC ≈ 0.49 and 0.47 — statistically indistinguishable from 0.5 (random guessing).** This is the
   clearest signal in the results: the models' predicted probabilities don't rank real videos above fake
   ones any better than chance. Combined with real-class recall of 0.09 for both models, the evidence points
   to both models having learned to predict "fake" most of the time (matching the ~4.2:1 training class
   prior) rather than learning features that actually discriminate real from AI-generated content.

**CNN+LSTM did not outperform the CNN baseline** on any test metric — a valid, reportable finding (the
notebook's §11.1 prompt explicitly anticipates this outcome). Most likely explanation given the setup: the
LSTM's best checkpoint came from epoch 1 of only 4 epochs trained before early stopping, so it barely got
more optimization than the baseline despite having more capacity and a harder optimization landscape (LSTM
gates on top of the CNN features).

**Likely root cause: class imbalance, not model capacity.** With only 55 real training videos against 227
fake, and no class weighting in `BCEWithLogitsLoss` (no `pos_weight` set) or oversampling of the real class,
there's little pressure on either model to learn real-specific features — predicting "fake" by default is
already a low-loss strategy on this training distribution. This, not the CNN-vs-LSTM architecture choice,
is the dominant factor limiting both models here.

## Suggestions if you continue iterating (not applied — for your own follow-up)

- Set `pos_weight` in `BCEWithLogitsLoss` (e.g. weight the real class ~4x) or oversample real videos each epoch, to counter the class prior directly.
- Add face detection/cropping as preprocessing (already noted as future work in §10) — raw frames may include background regions diluting the signal.
- Treat the current ROC-AUC ≈ 0.5 result as a signal to fix class imbalance *before* spending more time on LSTM hyperparameters — right now neither model has cleared the "better than random ranking" bar.

## Files in this folder

- `executed_notebook.ipynb` — full notebook re-run with all real outputs (plots included)
- `metrics_summary.json` — structured version of the numbers above
- `classification_reports.txt` — sklearn classification reports for both models
- `misclassified_list.txt` — all 13 CNN+LSTM test errors with predicted probabilities
- `confusion_matrices.png`, `training_curves.png`, `misclassified_example.png`
- `baseline_best.pt`, `lstm_best.pt` — trained weights for this run
- `progress_log.txt` — timestamped execution log
