"""Real-time recognition with webcam or video file.
   Webcam:  python live.py
   Video:   python live.py myvideo.mp4        (press q to quit)"""
import sys
import cv2
from PIL import Image
from common import load_classifier, recognize

model, classes = load_classifier()
source = sys.argv[1] if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(source)

SCALE = 0.5        # shrink frame for faster detection
EVERY = 3          # run the models every 3rd frame, reuse result in between
results, n = [], 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if n % EVERY == 0:
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
        results = recognize(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)), model, classes)
    n += 1
    for box, name, conf in results:
        x1, y1, x2, y2 = [int(v / SCALE) for v in box]
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {conf:.0%}", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Face Recognition (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
