import cv2
import numpy as np

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def UpdateHSVColor(keycode, hsv_color):
    if ord('q') == keycode:
        hsv_color[0] += 1 if (hsv_color[0] + 1) <= 255 else 0
    elif ord('w') == keycode:
        hsv_color[1] += 1 if (hsv_color[1] + 1) <= 255 else 0
    elif ord('e') == keycode:
        hsv_color[2] += 1 if (hsv_color[2] + 1) <= 255 else 0
    elif ord('a') == keycode:
        hsv_color[0] -= 1 if (hsv_color[0] - 1) >= 0 else 0
    elif ord('s') == keycode:
        hsv_color[1] -= 1 if (hsv_color[1] - 1) >= 0 else 0
    elif ord('d') == keycode:
        hsv_color[2] -= 1 if (hsv_color[2] - 1) >= 0 else 0
    elif ord('r') == keycode:
        hsv_color[3] += 1 if (hsv_color[3] + 1) <= 255 else 0
    elif ord('t') == keycode:
        hsv_color[4] += 1 if (hsv_color[4] + 1) <= 255 else 0
    elif ord('y') == keycode:
        hsv_color[5] += 1 if (hsv_color[5] + 1) <= 255 else 0
    elif ord('f') == keycode:
        hsv_color[3] -= 1 if (hsv_color[3] - 1) >= 0 else 0
    elif ord('g') == keycode:
        hsv_color[4] -= 1 if (hsv_color[4] - 1) >= 0 else 0
    elif ord('h') == keycode:
        hsv_color[5] -= 1 if (hsv_color[5] - 1) >= 0 else 0
    else:
        return False
    return True

hsv_color = [20, 0, 0, 130, 214, 145]
while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = (hsv_color[0], hsv_color[1], hsv_color[2])
    upper_green = (hsv_color[3], hsv_color[4], hsv_color[5])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # mask_green = cv2.morphologyEx(mask_green, op=cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8), iterations=2)
    # mask_green = cv2.dilate(mask_green, kernel=np.ones((3, 3), np.uint8), iterations=1)

    keycode = cv2.waitKey(30)
    if UpdateHSVColor(keycode, hsv_color):
        print('lower({}), upper({})\n'.format(hsv_color[:3], hsv_color[3:]))

    cv2.imshow("Camera", frame)
    cv2.imshow("mask", mask_green)

capture.release()
cv2.destroyAllWindows()