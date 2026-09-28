"""
Standalone script replicating the "sort into real/fake" step that Section 3 of the
notebook expects but doesn't actually contain (the notebook's Section 3 was written
for FaceForensics++, which ships pre-sorted; DFDC ships one folder + metadata.json).

Reads train_sample_videos/metadata.json and copies each video into data/real or
data/fake based on its label, matching the DATA_DIR/real, DATA_DIR/fake layout the
notebook's Section 3 sanity-check cell expects.
"""
import json
import shutil
from pathlib import Path

SRC_DIR = Path("train_sample_videos")
DATA_DIR = Path("data")

metadata = json.load(open(SRC_DIR / "metadata.json"))

(DATA_DIR / "real").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "fake").mkdir(parents=True, exist_ok=True)

counts = {"REAL": 0, "FAKE": 0}
skipped = []

for filename, info in metadata.items():
    label = info["label"]
    src = SRC_DIR / filename
    dest_folder = "real" if label == "REAL" else "fake"
    dest = DATA_DIR / dest_folder / filename

    if not src.exists():
        skipped.append(filename)
        continue
    if not dest.exists():
        shutil.copy2(src, dest)
    counts[label] += 1

print(f"Real videos copied: {counts['REAL']}")
print(f"Fake videos copied: {counts['FAKE']}")
if skipped:
    print(f"Skipped (source missing): {len(skipped)} -> {skipped[:10]}")
