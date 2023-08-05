from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# url = 'http://192.168.1.2:4747/video'  # redmi
url = 'http://192.168.125.40:4747/video'  # oppo

# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

cap = cv2.VideoCapture("./Videos/traffic2.mp4")

model = YOLO('yolo_weight/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "Laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("car_count/mask5.png")
mask = cv2.resize(mask, (640, 480))

# Tracking

tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

limits = [110, 380, 560, 380]

totalCount = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    imgRegion = cv2.bitwise_and(frame, mask)

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # cv2.rectangle(frame, (x1, y1), (x2, y2),
            #               (0, 255, 0), 2, cv2.LINE_4)

            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # if currentClass == "car" or currentClass == "bus" or currentClass == "truck" or currentClass == "motorbike" and conf > 0.3:
            if currentClass == "car" and conf > 0.3:
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=10, t=2)
                cvzone.putTextRect(frame, f"{currentClass} {conf}", (max(
                    0, x1), max(45, y1)), scale=1, thickness=1, offset=3)

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    result_tracker = tracker.update(detections)

    cv2.line(frame, (limits[0], limits[1]),
             (limits[2], limits[3]), (0, 0, 255), 3, cv2.LINE_4)

    for res in result_tracker:
        x1, y1, x2, y2, id = map(int, res)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, t=2,
                          rt=1, colorC=(0, 255, 0))
        # cvzone.putTextRect(frame, f"{id}", (max(
        #     0, x1), max(35, y1)), scale=2, thickness=2, offset=6)

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(frame, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-15 < cy < limits[1]+15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(frame, (limits[0], limits[1]),
                         (limits[2], limits[3]), (0, 255, 0), 3, cv2.LINE_4)

    cvzone.putTextRect(
        frame, f"Count: {len(totalCount)}", (20, 30), scale=2, thickness=2, offset=8)

    cv2.imshow("Car Counter", frame)
    # cv2.imshow("Webcam im", imgRegion)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
