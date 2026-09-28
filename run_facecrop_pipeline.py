"""
Autonomous overnight run: face-cropped, decord-decoded pipeline. Baseline + LSTM training,
tuning skipped (reusing best_cfg from 2026-09-09), evaluation, plots.

HARD DEADLINE: this must finish well before 06:00 on 2026-09-11 (user is asleep, no one to
rescue an overrun run). A wall-clock deadline is checked before every epoch of every stage;
if there isn't enough time left for another epoch (based on the slowest epoch seen so far,
plus a safety margin), that stage stops early and moves on, so the run always finishes cleanly
with real, saved results rather than getting killed mid-write.
"""
import os
import glob
import json
import random
import time
import datetime

import numpy as np
import torch  # must import before cv2/decord -- decord loads a DLL that otherwise
              # breaks torch's own DLL init on Windows (reproduced and confirmed)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import decord

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data"
NUM_FRAMES = 8
IMG_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 2
DROPOUT_P = 0.3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
EPOCHS_BASELINE = 8
EPOCHS_LSTM = 8
BEST_CFG = {"hidden_size": 128, "lr": 1e-4}
CHECKPOINT_DIR = "checkpoints"
OUT_DIR = "analysis/2026-09-11_v1"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Hard deadline: 2026-09-11 05:20 local time -- leaves ~40min safety margin before the
# user's actual 06:00 cutoff, to account for evaluation/plotting/saving after training stops.
HARD_DEADLINE = datetime.datetime(2026, 9, 11, 5, 20, 0).timestamp()

t_start = time.time()
progress_log_path = os.path.join(OUT_DIR, "progress_log.txt")
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(progress_log_path, "a") as f:
        f.write(line + "\n")

log(f"Using device: {DEVICE}")
log(f"Hard deadline: {datetime.datetime.fromtimestamp(HARD_DEADLINE).strftime('%H:%M:%S')} "
    f"({(HARD_DEADLINE - time.time())/3600:.2f} hours from now)")

FACE_MODEL_PATH = "face_detector_model/face_detection_yunet_2023mar.onnx"
face_detector = cv2.FaceDetectorYN.create(FACE_MODEL_PATH, "", (320, 320), score_threshold=0.6)

real_videos = sorted(glob.glob(os.path.join(DATA_DIR, "real", "*.mp4")))
fake_videos = sorted(glob.glob(os.path.join(DATA_DIR, "fake", "*.mp4")))
log(f"Real videos: {len(real_videos)} | Fake videos: {len(fake_videos)}")


def sample_frame_indices(total_frames, num_frames):
    if total_frames <= num_frames:
        return np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)
    return np.linspace(0, total_frames - 1, num_frames).astype(int)


def detect_face_box(frame_rgb):
    """Detect the highest-confidence face in an RGB frame, return a padded (x1,y1,x2,y2) box
    or None if no face found (caller falls back to the full frame in that case)."""
    h, w = frame_rgb.shape[:2]
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame_bgr)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[-1])
    x, y, bw, bh = best[0], best[1], best[2], best[3]
    x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
    pad_x, pad_y = int(bw * 0.3), int(bh * 0.3)
    x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def load_video_frames(video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    """Decode with decord (fast), detect+crop the face once on the middle sampled frame and
    reuse that same crop region for all frames of this clip (short clips, subject roughly
    static -- avoids 8x the detection cost for negligible accuracy loss). Falls back to the
    full frame if no face is detected, and to a black frame if the video can't be read at all."""
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        indices = sample_frame_indices(total_frames, num_frames).tolist()
        frames_arr = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) RGB uint8

        mid_frame = frames_arr[len(frames_arr) // 2]
        box = detect_face_box(mid_frame)

        frames = []
        for f in frames_arr:
            if box is not None:
                x1, y1, x2, y2 = box
                f = f[y1:y2, x1:x2]
            frames.append(cv2.resize(f, (img_size, img_size)))
    except Exception:
        frames = []

    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])
    if len(frames) == 0:
        frames = [np.zeros((img_size, img_size, 3), dtype=np.uint8)] * num_frames
    return np.stack(frames[:num_frames])


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
train_transform = T.Compose([
    T.ToPILImage(), T.RandomHorizontalFlip(p=0.5), T.ColorJitter(brightness=0.15, contrast=0.15),
    T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
eval_transform = T.Compose([
    T.ToPILImage(), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class VideoFrameDataset(Dataset):
    def __init__(self, video_paths, labels, transform):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path = self.video_paths[idx]
        label = self.labels[idx]
        frames = load_video_frames(path)
        transformed = [self.transform(f) for f in frames]
        clip = torch.stack(transformed, dim=0)
        return clip, torch.tensor(label, dtype=torch.float32), path


def split_videos(video_paths, labels, val_split=VAL_SPLIT, test_split=TEST_SPLIT, seed=SEED):
    class_indices = {}
    for i, label in enumerate(labels):
        class_indices.setdefault(label, []).append(i)
    train_idx, val_idx, test_idx = [], [], []
    for label, indices in class_indices.items():
        shuffled = indices[:]
        random.Random(seed).shuffle(shuffled)
        n = len(shuffled)
        n_test = int(n * test_split); n_val = int(n * val_split)
        test_idx.extend(shuffled[:n_test])
        val_idx.extend(shuffled[n_test:n_test + n_val])
        train_idx.extend(shuffled[n_test + n_val:])

    def subset(indices):
        return [video_paths[i] for i in indices], [labels[i] for i in indices]

    return subset(train_idx), subset(val_idx), subset(test_idx)


all_paths = real_videos + fake_videos
all_labels = [0] * len(real_videos) + [1] * len(fake_videos)
(train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_videos(all_paths, all_labels)
log(f"Train videos: {len(train_paths)} | Val videos: {len(val_paths)} | Test videos: {len(test_paths)}")

train_ds = VideoFrameDataset(train_paths, train_labels, transform=train_transform)
val_ds = VideoFrameDataset(val_paths, val_labels, transform=eval_transform)
test_ds = VideoFrameDataset(test_paths, test_labels, transform=eval_transform)

n_real_train, n_fake_train = train_labels.count(0), train_labels.count(1)
class_count = {0: n_real_train, 1: n_fake_train}
train_sample_weights = [1.0 / class_count[label] for label in train_labels]
train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_labels), replacement=True)
POS_WEIGHT = n_real_train / n_fake_train
log(f"Train class counts -> real: {n_real_train}, fake: {n_fake_train} | POS_WEIGHT: {POS_WEIGHT:.4f}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


class CNNBaselineClassifier(nn.Module):
    def __init__(self, pretrained=True, dropout_p=DROPOUT_P):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        backbone.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(backbone.fc.in_features, 1))
        self.backbone = backbone

    def forward(self, x):
        B, Tn, C, H, W = x.shape
        x = x.view(B * Tn, C, H, W)
        frame_logits = self.backbone(x).view(B, Tn)
        return frame_logits.mean(dim=1)


class CNNLSTMClassifier(nn.Module):
    def __init__(self, hidden_size=256, num_lstm_layers=1, dropout_p=DROPOUT_P, pretrained=True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.cnn = backbone
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size,
                             num_layers=num_lstm_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        B, Tn, C, H, W = x.shape
        x = x.view(B * Tn, C, H, W)
        features = self.cnn(x).view(B, Tn, self.feature_dim)
        lstm_out, (h_n, c_n) = self.lstm(features)
        final_hidden = self.dropout(h_n[-1])
        return self.classifier(final_hidden).squeeze(1)


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for clips, labels, _ in tqdm(loader, leave=False):
            clips, labels = clips.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            logits = model(clips)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * clips.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend((probs > 0.5).astype(int).tolist())
            all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    return avg_loss, acc


def train_model(model, train_loader, val_loader, epochs, lr, checkpoint_path, patience=3,
                 pos_weight=None, min_epochs_before_checkpoint=2, tag=""):
    model.to(DEVICE)
    train_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=DEVICE)) if pos_weight else nn.BCEWithLogitsLoss()
    val_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = None
    epoch_durations = []

    for epoch in range(1, epochs + 1):
        # Hard deadline guard: if the slowest epoch so far wouldn't fit before the deadline
        # (with a margin), stop this stage now rather than risk overrunning.
        if epoch_durations:
            worst = max(epoch_durations)
            if time.time() + worst * 1.15 > HARD_DEADLINE:
                log(f"[{tag}] Stopping before epoch {epoch}: not enough time left before the "
                    f"{datetime.datetime.fromtimestamp(HARD_DEADLINE).strftime('%H:%M:%S')} deadline "
                    f"(worst epoch so far took {worst:.0f}s).")
                break

        ep_t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, train_criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, val_criterion, optimizer=None)
        ep_duration = time.time() - ep_t0
        epoch_durations.append(ep_duration)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        log(f"[{tag}] Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {ep_duration:.1f}s")

        eligible = epoch >= min_epochs_before_checkpoint
        if eligible and val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        elif eligible:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log(f"[{tag}] Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                break

    if not os.path.exists(checkpoint_path):
        torch.save(model.state_dict(), checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))
    log(f"[{tag}] Best checkpoint: epoch {best_epoch}")
    return model, history, best_epoch


def get_predictions(model, loader):
    model.eval()
    all_probs, all_preds, all_labels, all_paths = [], [], [], []
    with torch.no_grad():
        for clips, labels, paths in tqdm(loader, leave=False):
            clips = clips.to(DEVICE)
            logits = model(clips)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().astype(int).tolist())
            all_paths.extend(paths)
    return np.array(all_probs), np.array(all_preds), np.array(all_labels), all_paths


def compute_metrics(labels, preds, probs, name):
    metrics = {
        "model": name,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


if __name__ == "__main__":
    log("=== Training CNN baseline (face-cropped pipeline) ===")
    baseline_model = CNNBaselineClassifier(pretrained=True)
    baseline_model, baseline_history, baseline_best_epoch = train_model(
        baseline_model, train_loader, val_loader,
        epochs=EPOCHS_BASELINE, lr=1e-4,
        checkpoint_path=os.path.join(CHECKPOINT_DIR, "baseline_best_2026-09-11.pt"),
        pos_weight=POS_WEIGHT, tag="baseline",
    )

    log(f"=== Training CNN+LSTM (face-cropped pipeline, reusing best_cfg={BEST_CFG}) ===")
    lstm_model = CNNLSTMClassifier(hidden_size=BEST_CFG["hidden_size"])
    lstm_model, lstm_history, lstm_best_epoch = train_model(
        lstm_model, train_loader, val_loader,
        epochs=EPOCHS_LSTM, lr=BEST_CFG["lr"],
        checkpoint_path=os.path.join(CHECKPOINT_DIR, "lstm_best_2026-09-11.pt"),
        pos_weight=POS_WEIGHT, tag="lstm",
    )

    log("=== Evaluating on test set ===")
    baseline_probs, baseline_preds, test_labels_b, test_paths_b = get_predictions(baseline_model, test_loader)
    lstm_probs, lstm_preds, test_labels_l, test_paths_l = get_predictions(lstm_model, test_loader)

    baseline_metrics = compute_metrics(test_labels_b, baseline_preds, baseline_probs, "CNN Baseline")
    lstm_metrics = compute_metrics(test_labels_l, lstm_preds, lstm_probs, "CNN + LSTM")
    log(f"Baseline metrics: {baseline_metrics}")
    log(f"LSTM metrics: {lstm_metrics}")

    with open(os.path.join(OUT_DIR, "classification_reports.txt"), "w") as f:
        f.write("=== CNN Baseline - classification report ===\n")
        f.write(classification_report(test_labels_b, baseline_preds, target_names=["real", "fake"]))
        f.write("\n=== CNN + LSTM - classification report ===\n")
        f.write(classification_report(test_labels_l, lstm_preds, target_names=["real", "fake"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, preds, name in zip(axes, [baseline_preds, lstm_preds], ["CNN Baseline", "CNN + LSTM"]):
        cm = confusion_matrix(test_labels_b, preds)
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "fake"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["real", "fake"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(baseline_history["train_loss"], label="Baseline train")
    axes[0].plot(baseline_history["val_loss"], label="Baseline val")
    axes[0].plot(lstm_history["train_loss"], label="CNN+LSTM train")
    axes[0].plot(lstm_history["val_loss"], label="CNN+LSTM val")
    axes[0].set_title("Loss curves"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[1].plot(baseline_history["train_acc"], label="Baseline train")
    axes[1].plot(baseline_history["val_acc"], label="Baseline val")
    axes[1].plot(lstm_history["train_acc"], label="CNN+LSTM train")
    axes[1].plot(lstm_history["val_acc"], label="CNN+LSTM val")
    axes[1].set_title("Accuracy curves"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=120)
    plt.close(fig)

    misclassified_idx = np.where(lstm_preds != test_labels_l)[0]
    with open(os.path.join(OUT_DIR, "misclassified_list.txt"), "w") as f:
        f.write(f"CNN+LSTM misclassified {len(misclassified_idx)} / {len(test_labels_l)} test clips.\n")
        for i in misclassified_idx:
            true_label = "fake" if test_labels_l[i] == 1 else "real"
            pred_label = "fake" if lstm_preds[i] == 1 else "real"
            f.write(f"- {os.path.basename(test_paths_l[i])} | true={true_label} pred={pred_label} "
                    f"prob_fake={lstm_probs[i]:.3f}\n")

    summary = {
        "run": "2026-09-11_v1",
        "pipeline": "decord decode + YuNet face-crop (once per video) + checkpoint-fix + tuning skipped",
        "dataset": {"real": len(real_videos), "fake": len(fake_videos), "total": len(all_paths)},
        "train_val_test": {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)},
        "best_cfg_reused": BEST_CFG,
        "baseline": {**baseline_metrics, "best_epoch": baseline_best_epoch, "history": baseline_history},
        "lstm": {**lstm_metrics, "best_epoch": lstm_best_epoch, "history": lstm_history},
        "total_runtime_seconds": time.time() - t_start,
    }
    with open(os.path.join(OUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log(f"=== DONE. Total runtime: {(time.time()-t_start)/3600:.2f} hours ===")
