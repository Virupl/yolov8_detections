import cvzone
import cv2
from ultralytics import YOLO
import math
from sort import *

video = "./Videos/people.mp4"

model = YOLO('yolo_weight/yolov8n.pt')

cap = cv2.VideoCapture(video)

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "Laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("people_count/mask.png")
mask = cv2.resize(mask, (640, 480))


# Tracking

tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)

limitsUp = [76, 161, 164, 161]
limitsDown = [283, 300, 370, 300]

upCount = []
downCount = []


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    imgRegion = cv2.bitwise_and(frame, mask)

    imgGraphic = cv2.imread("people_count/graphic.png", cv2.IMREAD_UNCHANGED)

    frame = cvzone.overlayPNG(frame, imgGraphic, (438, 0))

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

            if currentClass == "person" and conf > 0.3:
                # cvzone.cornerRect(frame, (x1, y1, w, h), l=10, t=2, rt=1)
                cvzone.putTextRect(
                    frame, f"{currentClass} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    result_tracker = tracker.update(detections)

    cv2.line(frame, (limitsUp[0], limitsUp[1]),
             (limitsUp[2], limitsUp[3]), (0, 0, 255), 3, cv2.LINE_4)

    cv2.line(frame, (limitsDown[0], limitsDown[1]),
             (limitsDown[2], limitsDown[3]), (0, 0, 255), 3, cv2.LINE_4)

    for res in result_tracker:
        x1, y1, x2, y2, id = map(int, res)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(frame, (x1, y1, w, h), l=5, t=2,
                          rt=1, colorC=(0, 255, 0))
        # cvzone.putTextRect(frame, f"{id}", (max(
        #     0, x1), max(35, y1)), scale=2, thickness=2, offset=6)

        cx, cy = x1+w//2, y1+h//2
        cv2.circle(frame, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1]-15 < cy < limitsUp[3]+15:
            if upCount.count(id) == 0:
                upCount.append(id)
                cv2.line(frame, (limitsUp[0], limitsUp[1]),
                         (limitsUp[2], limitsUp[3]), (0, 255, 0), 3, cv2.LINE_4)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1]-15 < cy < limitsDown[3]+15:
            if downCount.count(id) == 0:
                downCount.append(id)
                cv2.line(frame, (limitsDown[0], limitsDown[1]),
                         (limitsDown[2], limitsDown[3]), (0, 255, 0), 3, cv2.LINE_4)

    cv2.putText(
        frame, str(len(upCount)), (495, 42), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.putText(
        frame, str(len(downCount)), (595, 42), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("People Counter", frame)
    # cv2.imshow("Webcam im", imgRegion)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
