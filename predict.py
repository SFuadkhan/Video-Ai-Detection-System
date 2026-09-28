"""
Run the trained deepfake detector on any video file.

Usage:
    venv\\Scripts\\python.exe predict.py path\\to\\video.mp4
    venv\\Scripts\\python.exe predict.py path\\to\\video.mp4 --model baseline

Uses the CNN+LSTM model by default (the final, best-performing model from
analysis/2026-09-11_v2 -- 90.4% test accuracy, 0.964 ROC-AUC). Pass --model baseline
to use the simpler CNN baseline instead (87.5% test accuracy, 0.928 ROC-AUC).

Runs the exact same preprocessing used during training: decord decode, YuNet
face-crop (once per clip), resize, ImageNet normalization -- so predictions are
consistent with the reported results.
"""
import argparse
import os
import sys
import urllib.request

import numpy as np
import torch  # must import before cv2/decord -- decord loads a DLL that otherwise
              # breaks torch's own DLL init on Windows (reproduced and confirmed
              # during this project)
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import cv2
import decord

NUM_FRAMES = 8
IMG_SIZE = 224
DROPOUT_P = 0.3
FACE_MODEL_PATH = "face_detector_model/face_detection_yunet_2023mar.onnx"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
eval_transform = T.Compose([
    T.ToPILImage(), T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

FACE_MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
                  "face_detection_yunet/face_detection_yunet_2023mar.onnx")
if not os.path.exists(FACE_MODEL_PATH):
    # The model file is gitignored, so fetch it on first run (same as the notebook does).
    os.makedirs(os.path.dirname(FACE_MODEL_PATH), exist_ok=True)
    print(f"Downloading YuNet face detector to {FACE_MODEL_PATH} ...")
    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)

face_detector = cv2.FaceDetectorYN.create(FACE_MODEL_PATH, "", (320, 320), score_threshold=0.6)


def sample_frame_indices(total_frames, num_frames):
    if total_frames <= num_frames:
        return np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)
    return np.linspace(0, total_frames - 1, num_frames).astype(int)


def detect_face_box(frame_rgb):
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
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def load_video_frames(video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total_frames = len(vr)
    indices = sample_frame_indices(total_frames, num_frames).tolist()
    frames_arr = vr.get_batch(indices).asnumpy()

    mid_frame = frames_arr[len(frames_arr) // 2]
    box = detect_face_box(mid_frame)
    face_found = box is not None

    frames = []
    for f in frames_arr:
        if box is not None:
            x1, y1, x2, y2 = box
            f = f[y1:y2, x1:x2]
        frames.append(cv2.resize(f, (img_size, img_size)))
    return np.stack(frames), face_found


class CNNBaselineClassifier(nn.Module):
    def __init__(self, pretrained=False, dropout_p=DROPOUT_P):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(backbone.fc.in_features, 1))
        self.backbone = backbone

    def forward(self, x):
        B, Tn, C, H, W = x.shape
        x = x.view(B * Tn, C, H, W)
        frame_logits = self.backbone(x).view(B, Tn)
        return frame_logits.mean(dim=1)


class CNNLSTMClassifier(nn.Module):
    def __init__(self, hidden_size=256, num_lstm_layers=1, dropout_p=DROPOUT_P):
        super().__init__()
        backbone = models.resnet18(weights=None)
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


def main():
    parser = argparse.ArgumentParser(description="Run the trained deepfake detector on a video.")
    parser.add_argument("video_path", help="Path to an .mp4 video file")
    parser.add_argument("--model", choices=["lstm", "baseline"], default="lstm",
                         help="Which trained model to use (default: lstm, the best-performing one)")
    args = parser.parse_args()

    print(f"Loading {args.model} model...")
    if args.model == "lstm":
        model = CNNLSTMClassifier(hidden_size=256)
        checkpoint_path = "checkpoints/lstm_best_2026-09-11_v2.pt"
    else:
        model = CNNBaselineClassifier()
        checkpoint_path = "checkpoints/baseline_best_2026-09-11_v2.pt"

    if not os.path.exists(checkpoint_path):
        sys.exit(f"Checkpoint not found: {checkpoint_path}\n"
                 "The final weights ship with the repository in checkpoints/. Re-clone or "
                 "restore that folder, or retrain with run_facecrop_pipeline_v2_tuned.py "
                 "(see README.md).")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"Reading and face-cropping frames from: {args.video_path}")
    frames, face_found = load_video_frames(args.video_path)
    if not face_found:
        print("WARNING: no face detected -- falling back to full (uncropped) frames. "
              "Prediction may be less reliable.")

    transformed = torch.stack([eval_transform(f) for f in frames], dim=0)  # (T, C, H, W)
    clip = transformed.unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)

    with torch.no_grad():
        logit = model(clip)
        prob_fake = torch.sigmoid(logit).item()

    verdict = "FAKE" if prob_fake > 0.5 else "REAL"
    print()
    print(f"Prediction: {verdict}")
    print(f"Probability fake: {prob_fake:.3f}  (probability real: {1 - prob_fake:.3f})")


if __name__ == "__main__":
    main()
