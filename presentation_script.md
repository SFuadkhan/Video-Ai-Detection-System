# Presentation Script — AI-Generated Video Detection Using CNN Features and LSTM-Based Temporal Modeling

Target length: ~12-13 minutes at a natural speaking pace. Read through once beforehand and adjust
to your own voice — this is a starting point, not a script to memorize word for word.

---

## 1. Introduction & Problem (≈1.5 min)

Hi, I'm Fuadkhan Safarov, and this is my final project for Deep Learning: detecting AI-generated —
"deepfake" — videos using a custom CNN and LSTM-based architecture.

The problem: deepfake videos, where a person's face is digitally swapped or manipulated to make
them appear to say or do something they never did, are becoming a real, growing problem. They're
used for misinformation, fraud, and harassment, and they're getting easier to produce and harder to
spot with the naked eye. Automated detection is a genuinely important, real-world application of
deep learning.

My specific approach compares two models: a straightforward frame-by-frame CNN baseline, and a
CNN+LSTM model that adds temporal reasoning across frames. The core question I wanted to answer
wasn't just "can I build a detector" — it was "does modeling how a video changes over time actually
help detect fakes better than just looking at individual frames?"

## 2. Dataset & Preprocessing (≈2 min)

For data, I used a combination of two sources: a sample from the DFDC — the Deepfake Detection
Challenge — dataset, and Google/Jigsaw's DFD dataset, both real face-swap deepfake datasets, not
synthetic or AI-generated content in the broader sense.

One real challenge here: the original DFDC competition on Kaggle turned out to be closed to new
entrants, so I couldn't download the full dataset that way. I pivoted to a different, ungated Kaggle
Dataset — DFD — which gave me real actors and their face-swapped counterparts without needing
competition access. In total, after merging and balancing, I trained on 1,390 videos: 440 real, 950
fake.

For preprocessing: from each video I sample 8 evenly-spaced frames. Each frame gets resized to
224x224 and normalized using standard ImageNet statistics, since both models use an
ImageNet-pretrained backbone. For training data specifically, I apply light augmentation — random
horizontal flips and color jitter — to help the models generalize rather than memorize specific
lighting or framing.

The single most impactful preprocessing step, though, was face cropping. Early on, my pipeline fed
whole video frames — background included — to the model. Deepfake manipulation only touches the
face region, so a huge fraction of every frame was pixels that never actually differ between real
and fake. Once I added a face-detection-and-crop step before classification, results jumped
dramatically — I'll show the numbers shortly.

## 3. Model Architecture (≈2.5 min)

Both models share a ResNet18 backbone, pretrained on ImageNet, which gives a strong starting point
for visual feature extraction without needing to train a CNN from scratch on a relatively small
deepfake dataset.

The baseline model is intentionally simple: each of the 8 frames gets passed through the CNN
independently, producing a real-or-fake score per frame, and those 8 scores get averaged into one
final prediction for the clip. No memory of frame order, no temporal reasoning at all. This exists
specifically as a reference point — the bar the more sophisticated model has to beat.

The CNN+LSTM model uses the same CNN backbone, but instead of classifying each frame independently,
it extracts a 512-dimensional feature vector per frame and feeds that sequence into an LSTM layer.
The idea is that an LSTM can pick up on patterns across frames — subtle inconsistencies in how a
face moves or is lit over time — that a model looking at one frame at a time simply can't see. The
LSTM's final hidden state goes through dropout and a fully connected layer to produce the
prediction.

Both models use a single output logit with a sigmoid, trained with binary cross-entropy loss. I used
a weighted random sampler and a class-weighted loss function to handle the real-vs-fake class
imbalance in the training data, since fakes substantially outnumber real videos in this domain by
nature — one real video can generate many fakes.

## 4. Training, Tuning, and Challenges (≈3 min)

This is where a lot of the real engineering work happened, and I want to be upfront about it because
I think the debugging process is as valuable to show as the final numbers.

**First challenge — a training bug that looked like a modeling failure.** Early on, my LSTM model
scored dramatically worse than the baseline — 32% accuracy versus 64%. I could have concluded
"temporal modeling doesn't help here" and moved on. Instead, I dug into the training loop and found
a bug: my checkpoint-saving logic picked whichever epoch had the lowest validation loss, with no
minimum training period. On a small validation set, the very first, essentially untrained epoch
sometimes got a lucky-low loss purely from noise, and that under-trained snapshot got permanently
saved as "best." I fixed this by requiring at least two epochs of training before a checkpoint is
eligible to be considered the best one.

**Second challenge — a genuine performance bottleneck: video decoding speed.** My first full training
run on the larger dataset took over 7 hours. Profiling showed the actual bottleneck wasn't the GPU at
all — it was CPU-bound video decoding. My code was decoding entire 200-to-300-frame videos just to
keep 8 sampled frames. Switching from OpenCV to a purpose-built video-decoding library called decord
cut that same run down to under 2 hours, without changing any model logic.

**Third, and the most important improvement — face cropping**, which I mentioned already. Adding a
face-detection step before classification took my ROC-AUC score — a measure of how well the model
distinguishes real from fake — from around 0.7, barely better than chance, up to the 0.93-to-0.96
range.

**Fourth — proper hyperparameter tuning.** For my final training run, I ran a small tuning search —
three configurations varying the LSTM's hidden size and learning rate — actually on the final,
face-cropped pipeline, rather than reusing a configuration found earlier on a different setup. This
turned out to matter a lot, which I'll get to in the results.

I also hit two Windows-specific technical issues worth a quick mention: running my notebook through
one particular execution tool crashed unpredictably for reasons unrelated to my code, and importing
one video library before PyTorch caused a DLL conflict. Both are documented with fixes in the
notebook's comments in case anyone else runs into them.

## 5. Results (≈2.5 min)

Here are the final results, from the properly-tuned run on the full 1,390-video dataset. [Show
results table / confusion matrix / training curves here.]

The CNN baseline reached 87.5% accuracy with a ROC-AUC of 0.928. The CNN+LSTM model, with tuned
hyperparameters — a hidden size of 256 instead of the smaller 128 I'd been reusing — reached 90.4%
accuracy with a ROC-AUC of 0.964, outperforming the baseline on every single metric: accuracy,
precision, recall, F1, and ROC-AUC.

This is actually a really interesting part of the story. An earlier version of this exact same
pipeline — same data, same code, differing only in using an under-tuned LSTM configuration — found
the *baseline* winning instead. That wasn't a wrong result given its inputs; it correctly showed that
an under-provisioned LSTM loses to a simple baseline. Once I properly tuned the LSTM specifically for
this pipeline, the conclusion reversed. The takeaway: a temporal model is only as good as the tuning
behind it, and hyperparameter choices can flip your entire conclusion about whether an architecture
is worth the added complexity.

Looking at the errors the tuned LSTM still makes — about 9.6% of test clips — they're roughly evenly
split between false positives and false negatives, with a mild tendency to struggle on videos with
quieter delivery or outdoor lighting, which is a reasonable direction for future data collection.

## 6. Conclusions & Future Work (≈1.5-2 min)

To sum up: I built and compared two deepfake detection architectures, and through an iterative
process of finding and fixing real bugs — a training bug, a performance bottleneck, missing face
cropping, and an under-tuned model — went from results barely better than random guessing to a model
correctly detecting fake videos over 90% of the time, with genuine evidence that temporal modeling
adds real value once properly configured.

For future work, I'd want to: test generalization on a completely different dataset like Celeb-DF, to
see whether these results hold up on manipulation techniques the model has never seen; try replacing
the LSTM with a Transformer encoder to compare temporal modeling approaches; and run multiple seeds
to quantify how much of my reported numbers is genuine signal versus normal run-to-run variance,
which I did observe some of even between otherwise-identical runs.

Thanks for watching.

---

## Notes for recording

- Approximate section timings add up to about 12.5-13.5 minutes — comfortably inside the 10-15
  minute window even with natural pauses.
- Have the results table, confusion matrix, and training curves images visible on screen during
  Section 5 — that's the part a viewer most wants to actually see, not just hear about.
- It's fine to pause the recording between sections and stitch them together, or to do it in one
  take reading loosely from this script. Either is normal for this kind of submission.
- Windows has a built-in screen recorder: press Win+G to open the Xbox Game Bar, then use its
  Capture widget to record your screen (and microphone) directly. No extra software needed.
