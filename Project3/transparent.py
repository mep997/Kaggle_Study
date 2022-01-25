import cv2
import numpy as np

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

background = cv2.imread('background.jpg')

hsv_color = [20, 0, 0, 130, 214, 145]
while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = (hsv_color[0], hsv_color[1], hsv_color[2])
    upper_green = (hsv_color[3], hsv_color[4], hsv_color[5])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    mask_green = cv2.morphologyEx(mask_green, op=cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2)
    mask_green = cv2.dilate(mask_green, kernel=np.ones((3, 3), np.uint8), iterations=1)
    mask_target = cv2.bitwise_not(mask_green)

    frame_background = cv2.bitwise_and(background, background, mask=mask_green)
    frame_target = cv2.bitwise_and(frame, frame, mask=mask_target)
    frame_result = cv2.addWeighted(src1=frame_background, alpha=1, src2=frame_target, beta=1, gamma=0)

    # cv2.imshow('background', frame_background)
    # cv2.imshow('target', frame_target)
    cv2.imshow('result', frame_result)

    keycode = cv2.waitKey(30)
    if ord('q') == keycode:
        break

capture.release()
cv2.destroyAllWindows()