from ultralytics import YOLO
import cv2

model = YOLO('yolo_weight/yolov8n.pt')

# url = 'http://192.168.1.2:4747/video'  # redmi
url = 'http://192.168.125.40:4747/video'  # oppo

video = cv2.VideoCapture(url)
video.set(3, 640)
video.set(4, 480)

while True:
    ret, frame = video.read()

    if not ret:
        break

    # result = model("yolo_basic/child.jpg", show=True)
    result = model(frame, stream=True)

    cv2.imshow("Yolo webcame", frame)

    if cv2.waitKey(27) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
