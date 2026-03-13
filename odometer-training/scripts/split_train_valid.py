import os
import random
import shutil
from collections import defaultdict

BASE = "models/digit_detector/data_cropped"
TRAIN_IMG = f"{BASE}/train/images"
TRAIN_LBL = f"{BASE}/train/labels"
VAL_IMG = f"{BASE}/valid/images"
VAL_LBL = f"{BASE}/valid/labels"

os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)

groups = defaultdict(list)

for img in os.listdir(TRAIN_IMG):
    if not img.endswith(".jpg"):
        continue

    key = img.split(".jpg")[0].split(".rf")[0]
    groups[key].append(img)

keys = list(groups.keys())
random.shuffle(keys)

val_ratio = 0.2
val_count = int(len(keys) * val_ratio)
val_keys = set(keys[:val_count])

for key in val_keys:
    for img in groups[key]:
        lbl = img.replace(".jpg", ".txt")
        shutil.move(f"{TRAIN_IMG}/{img}", f"{VAL_IMG}/{img}")
        shutil.move(f"{TRAIN_LBL}/{lbl}", f"{VAL_LBL}/{lbl}")

print("✅ grouped train / valid split done")