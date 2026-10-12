# End-to-End Face Detection & Recognition (CPU only)

## Method
- **Detection:** MTCNN (deep learning, handles multiple faces / partial occlusion)
- **Recognition:** FaceNet (InceptionResnetV1, pretrained on VGGFace2) gives a 512-d embedding per face.
  A small neural network classifier is **trained by us** on those embeddings (transfer learning).
- **Unknown faces:** if confidence < 0.60, the face is labelled "Unknown".
- **Data:** 80/20 split per person, then augmentation (flip, rotation, brightness/contrast, zoom)
  applied to the **training set only** (so no leakage into the test set).

## Setup
    pip install -r requirements.txt
    pip install facenet-pytorch --no-deps

## Dataset (Kaggle)
Celebrity Face Image Dataset (18 people x 100 images). Unzip so that it looks like:
    data/raw/<Person Name>/*.jpg

## Run
    python step1_split.py      # 80/20 split
    python step2_embed.py      # detect + augment + embed   (slowest step)
    python step3_train.py      # train, accuracy plot, test report
    streamlit run app.py       # UI: upload image / webcam / accuracy plot
    python live.py             # real-time webcam (or: python live.py video.mp4)

## Outputs
accuracy.png (accuracy vs epochs), confusion_matrix.png, model.pt
