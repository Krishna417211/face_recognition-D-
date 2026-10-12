"""STEP 2: detect face -> (augment, train only) -> FaceNet embedding. Saves .npz files."""
import os, json
import numpy as np
from PIL import Image
from torchvision import transforms as T
from common import find_faces, crop_face, embed

AUG_COPIES = 4   # augmented copies per training face

# Augmentation: flip, rotation, brightness/contrast, scaling (zoom)
augment = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.4, contrast=0.3),
    T.RandomResizedCrop(160, scale=(0.8, 1.0)),
])

classes = sorted(os.listdir("data/train"))
json.dump(classes, open("classes.json", "w"))

for split in ("train", "test"):
    X, y, skipped = [], [], 0
    for label, person in enumerate(classes):
        folder = f"data/{split}/{person}"
        for f in os.listdir(folder):
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            boxes = find_faces(img)
            if len(boxes) == 0:
                skipped += 1
                continue
            box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))   # biggest face
            face = crop_face(img, box)
            versions = [face]
            if split == "train":                       # augment ONLY the training set
                versions += [augment(face) for _ in range(AUG_COPIES)]
            for e in embed(versions):
                X.append(e)
                y.append(label)
        print(f"[{split}] {person} done")
    np.savez(f"{split}_emb.npz", X=np.array(X, dtype=np.float32), y=np.array(y, dtype=np.int64))
    print(f"{split}: {len(X)} samples saved, {skipped} images had no face\n")
