"""Shared helpers: face detection (MTCNN), face embedding (FaceNet), classifier."""
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1

torch.set_num_threads(4)

THRESHOLD = 0.90          # confidence below this -> "Unknown"
MODEL_PATH = "model.pt"

detector = MTCNN(keep_all=True, min_face_size=40, device="cpu")        # finds faces
embedder = InceptionResnetV1(pretrained="vggface2").eval()              # face -> 512 numbers


def find_faces(img):
    """Return list of [x1, y1, x2, y2] boxes for every face in a PIL image."""
    boxes, _ = detector.detect(img)
    return [] if boxes is None else boxes


def crop_face(img, box, margin=20):
    x1, y1, x2, y2 = box
    return img.crop((x1 - margin, y1 - margin, x2 + margin, y2 + margin)).resize((160, 160))


def embed(faces):
    """List of 160x160 PIL faces -> numpy array (N, 512)."""
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in faces])
    x = (torch.from_numpy(arr).permute(0, 3, 1, 2) - 127.5) / 128.0
    with torch.no_grad():
        return embedder(x).numpy()


class Classifier(nn.Module):
    """Small network trained on top of the FaceNet embeddings."""
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_classifier():
    ck = torch.load(MODEL_PATH)
    model = Classifier(len(ck["classes"]))
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["classes"]


def recognize(img, model, classes):
    """PIL image -> list of (box, name, confidence)."""
    boxes = find_faces(img)
    if len(boxes) == 0:
        return []
    embs = embed([crop_face(img, b) for b in boxes])
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(embs)), dim=1)
    results = []
    for box, p in zip(boxes, probs):
        conf, idx = p.max(0)
        name = classes[int(idx)] if float(conf) >= THRESHOLD else "Unknown"
        results.append((box, name, float(conf)))
    return results


def draw(img, results):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for box, name, conf in results:
        d.rectangle([float(v) for v in box], outline="lime", width=3)
        d.text((float(box[0]), max(0.0, float(box[1]) - 14)), f"{name} {conf:.0%}", fill="lime")
    return img
