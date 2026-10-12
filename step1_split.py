"""STEP 1: split raw dataset 80% train / 20% test (separately for every person)."""
import os, shutil
from sklearn.model_selection import train_test_split

RAW = "data/raw"
IMG_EXT = (".jpg", ".jpeg", ".png")

# If the zip had an extra wrapper folder, go inside it
while True:
    items = os.listdir(RAW)
    subdirs = [i for i in items if os.path.isdir(os.path.join(RAW, i))]
    if len(subdirs) == 1 and not any(i.lower().endswith(IMG_EXT) for i in items):
        RAW = os.path.join(RAW, subdirs[0])
    else:
        break

for split in ("train", "test"):
    shutil.rmtree(f"data/{split}", ignore_errors=True)

total = 0
for person in sorted(os.listdir(RAW)):
    folder = os.path.join(RAW, person)
    if not os.path.isdir(folder):
        continue
    files = [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)]
    if len(files) < 5:
        continue
    train, test = train_test_split(files, test_size=0.2, random_state=42)
    for split, names in (("train", train), ("test", test)):
        os.makedirs(f"data/{split}/{person}", exist_ok=True)
        for f in names:
            shutil.copy(os.path.join(folder, f), f"data/{split}/{person}/{f}")
    total += len(files)
    print(f"{person}: {len(train)} train, {len(test)} test")

print("Total images:", total)
