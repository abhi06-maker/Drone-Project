import cv2 as cv
import os
import time

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv.flip(frame, 1)
    cv.rectangle(frame, (200, 80), (456, 336), (0, 255, 0), 1)

    cv.imshow("Video Frame", frame)

    key_pressed = cv.waitKey(1) & 0xFF

    if key_pressed == ord('c'):
        img_crop = frame[81:335, 201:455]
        img_name = "img_{}.jpg".format(len(os.listdir()) + 1)
        cv.imwrite(img_name, img_crop)
        print("{} written!".format(img_name))
        time.sleep(2)

    if key_pressed == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
