import cv2
import dlib
import numpy as np
from imutils import resize

KEY_SPACE = 33
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Shape Predictor
def DrawFaceRect(frame, rect, color=COLOR_GREEN):
    return cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), color, 1)

def DrawFacePoint(frame, parts, color=COLOR_GREEN):
    for part in parts:
        pt = (part.x, part.y)
        frame = cv2.line(frame, pt, pt, color, 3)

    return frame

def DrawFaceLine(frame, parts, color=COLOR_GREEN):
    if len(parts) != 68:
        return frame

    def DrawFaceLineInner(frame, parts, idx_start, idx_end, isClosed=False):
        for idx in range(idx_start, idx_end):
            part1 = face.part(idx)
            pt1 = (part1.x, part1.y)
            part2 = face.part(idx+1)
            pt2 = (part2.x, part2.y)
            frame = cv2.line(frame, pt1, pt2, color, 1)

        if isClosed:
            part1 = face.part(idx_start)
            pt1 = (part1.x, part1.y)
            part2 = face.part(idx_end)
            pt2 = (part2.x, part2.y)
            frame = cv2.line(frame, pt1, pt2, color, 1)
        return frame

    # Face Shape
    frame = DrawFaceLineInner(frame, parts, 0, 16)
    # Eyebrows
    frame = DrawFaceLineInner(frame, parts, 17, 21)
    frame = DrawFaceLineInner(frame, parts, 22, 26)
    # Eyes
    frame = DrawFaceLineInner(frame, parts, 36, 41, isClosed=True)
    frame = DrawFaceLineInner(frame, parts, 42, 47, isClosed=True)
    # Nose
    frame = DrawFaceLineInner(frame, parts, 27, 30)
    frame = DrawFaceLineInner(frame, parts, 30, 35, isClosed=True)
    # mouth
    frame = DrawFaceLineInner(frame, parts, 48, 59, isClosed=True)
    frame = DrawFaceLineInner(frame, parts, 60, 67, isClosed=True)

    return frame

def GetFacePartImage(frame, parts, idx_start, idx_end, margin=(0, 0)):
    left = float('inf')
    top = float('inf')
    right = 0
    bottom = 0

    for idx in range(idx_start, idx_end + 1):
        part = face.part(idx)
        if left > part.x:
            left = part.x
        if top > part.y:
            top = part.y
        if right < part.x:
            right = part.x
        if bottom < part.y:
            bottom = part.y

    left = left-margin[0] if (left-margin[0]) >= 0 else 0
    top = top-margin[1] if (top-margin[1]) >= 0 else 0
    right = right+margin[0] if (right+margin[0]) <= frame.shape[0] else frame.shape[0]
    bottom = bottom+margin[1] if (bottom+margin[1]) <= frame.shape[1] else frame.shape[1]

    return frame[top:bottom, left:right].copy()

def DrawFacePartOnOrange(partImg, orangeImg, pt):
    try:
        orangeImg = cv2.seamlessClone(partImg, orangeImg, np.full(partImg.shape[:2], 255, partImg.dtype), pt, cv2.MIXED_CLONE)
    except:
        pass
    return orangeImg

detector = dlib.get_frontal_face_detector()
shape_preditor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
while cv2.waitKey(KEY_SPACE) < 0:
    orangeImg = cv2.imread('orange.jpg')
    orangeImg = cv2.resize(orangeImg, dsize=(512, 512))

    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    dets = detector(frame, 1)
    if len(dets) < 1:
        continue

    for det in dets:
        face = shape_preditor(frame, det)

        frame_LeftEye = GetFacePartImage(frame, face.parts(), 36, 41, margin=(3, 3))
        frame_RightEye = GetFacePartImage(frame, face.parts(), 42, 47, margin=(3, 3))
        frame_Mouth = GetFacePartImage(frame, face.parts(), 48, 59, margin=(3, 3))

        frame_LeftEye = resize(frame_LeftEye, width=100)
        frame_RightEye = resize(frame_RightEye, width=100)
        frame_Mouth = resize(frame_Mouth, width=150)

        orangeImg = DrawFacePartOnOrange(frame_LeftEye, orangeImg, (100, 180))
        orangeImg = DrawFacePartOnOrange(frame_RightEye, orangeImg, (250, 180))
        orangeImg = DrawFacePartOnOrange(frame_Mouth, orangeImg, (120, 320))

        # frame = DrawFaceRect(frame, face.rect, color=COLOR_GREEN)
        # frame = DrawFaceLine(frame, face.parts(), color=COLOR_BLUE)
        # frame = DrawFacePoint(frame, face.parts(), color=COLOR_RED)

        # cv2.imshow("Left Eye", frame_LeftEye)
        # cv2.imshow("Right Eye", frame_RightEye)
        # cv2.imshow("Mouth", frame_Mouth)

    # cv2.imshow("Camera", frame)
    cv2.imshow("Orange", orangeImg)

capture.release()
cv2.destroyAllWindows()