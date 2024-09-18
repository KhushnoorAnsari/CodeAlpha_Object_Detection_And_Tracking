import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('best.pt')
cap = cv2.VideoCapture('surf.mp4')

with open("Coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

count = 0
tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    frame = cv2.flip(frame, 1)
    results = model.predict(frame)

    # Extract data from results
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()  # Convert to numpy array
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    # Create a DataFrame
    data = np.hstack((xyxy, conf[:, np.newaxis], cls[:, np.newaxis]))
    columns = ['x1', 'y1', 'x2', 'y2', 'conf', 'cls']
    px = pd.DataFrame(data, columns=columns)

    list = []

    for index, row in px.iterrows():
        x1 = int(row['x1'])
        y1 = int(row['y1'])
        x2 = int(row['x2'])
        y2 = int(row['y2'])
        d = int(row['cls'])
        c = class_list[d]
        list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x4, y4, x5, y5, id = bbox
        cv2.rectangle(frame, (x4, y4), (x5, y5), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (x5, y5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
