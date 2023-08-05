from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO('yolo_weight/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "Laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# url = 'http://192.168.1.2:4747/video'  # redmi
url = 'http://192.168.125.40:4747/video'  # oppo

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # cv2.rectangle(frame, (x1, y1), (x2, y2),
            #               (0, 255, 0), 2, cv2.LINE_4)

            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))

            # Confidence
            conf = math.ceil(box.conf[0] * 100) / 100

            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(
                frame, f"{classNames[cls]} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(27) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
