import cv2
from ultralytics import YOLO
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO('best.pt')
classNames = ["bottle", "tao kae noi", "plastic cup"]


while True:
    success, img= cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence------>", confidence)

            cls= int(box.cls[0])
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            thickness = 2
            color = (0,0,0)

            if classNames[cls] == "bottle":
                color = (0, 255, 255)
                print("Recycle (Yellow)", classNames[cls])
                cv2.rectangle(img , (x1, y1), (x2, y2), color , 3)
                cv2.putText(img, "Recycle (yellow can)", org, font, fontScale, color, thickness)

            elif classNames[cls] == 'plastic cup':
                 color = (0, 255, 255)
                 print("Recycle (Yellow)", classNames[cls])
                 cv2.rectangle(img , (x1, y1), (x2, y2), color, 3)
                 cv2.putText(img, "Recycle (yellow can)", org, font, fontScale, color, thickness)

            elif classNames[cls] == 'tao kae noi':
                 print("General (Blue)", classNames[cls])
                 color = (255, 0, 0)
                 cv2.rectangle(img , (x1, y1), (x2, y2), color, 3)
                 cv2.putText(img, "General (blue can)", org, font, fontScale, color, thickness)


        

    cv2.imshow("Cam", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()