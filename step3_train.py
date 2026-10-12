"""STEP 3: train classifier on embeddings, plot accuracy vs epochs, evaluate on test set."""
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from common import Classifier, MODEL_PATH

EPOCHS, BATCH = 40, 64

classes = json.load(open("classes.json"))
tr, te = np.load("train_emb.npz"), np.load("test_emb.npz")
Xtr, ytr = torch.from_numpy(tr["X"]), torch.from_numpy(tr["y"])
Xte, yte = torch.from_numpy(te["X"]), torch.from_numpy(te["y"])

model = Classifier(len(classes))
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()


def accuracy(X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


train_acc, test_acc = [], []
for epoch in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), BATCH):
        idx = perm[i:i + BATCH]
        if len(idx) < 2:
            continue
        opt.zero_grad()
        loss_fn(model(Xtr[idx]), ytr[idx]).backward()
        opt.step()
    train_acc.append(accuracy(Xtr, ytr))
    test_acc.append(accuracy(Xte, yte))
    print(f"Epoch {epoch:2d}/{EPOCHS}  train acc {train_acc[-1]:.3f}  test acc {test_acc[-1]:.3f}")

torch.save({"state": model.state_dict(), "classes": classes}, MODEL_PATH)

# --- Accuracy vs epochs plot ---
plt.figure(figsize=(7, 4))
plt.plot(range(1, EPOCHS + 1), train_acc, label="Train accuracy")
plt.plot(range(1, EPOCHS + 1), test_acc, label="Test accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Face recognition accuracy")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("accuracy.png", dpi=150)

# --- Final test evaluation ---
model.eval()
with torch.no_grad():
    pred = model(Xte).argmax(1).numpy()
print(f"\nFINAL TEST ACCURACY: {accuracy(Xte, yte):.2%}\n")
print(classification_report(yte.numpy(), pred, target_names=classes))

fig, ax = plt.subplots(figsize=(9, 9))
ConfusionMatrixDisplay.from_predictions(yte.numpy(), pred, display_labels=classes,
                                        xticks_rotation=90, ax=ax, colorbar=False)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
print("Saved: model.pt, accuracy.png, confusion_matrix.png")
