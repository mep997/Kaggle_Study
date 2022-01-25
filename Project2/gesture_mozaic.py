import cv2
import mediapipe as mp
import numpy as np

KEY_SPACE = 33
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

gesture = {
    0:'False', 1:'True'
}

# Gesture recognition model
file = np.genfromtxt('custom_train.csv', delimiter=',')
x_train = file[1:,:-1].astype(np.float32)
y_train = file[1:, -1].astype(np.int)

classifier = cv2.ml.NormalBayesClassifier_create()
classifier.train(x_train, cv2.ml.ROW_SAMPLE, y_train)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hands = mp.solutions.hands.Hands()
while cv2.waitKey(KEY_SPACE) < 0:
    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(COLOR_RED),
                mp.solutions.drawing_utils.DrawingSpec(COLOR_GREEN))
                #mp.solutions.drawing_styles.get_default_hand_landmarks_style(), # point
                #mp.solutions.drawing_styles.get_default_hand_connections_style()) # line

            joint = np.zeros((21, 3))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            # Compute angles between joints
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]
            # 손 끝 마디 쪽은 계산하지 않음
            angle = np.degrees(angle)  # Convert radian to degree

            test_x = np.array([angle], dtype=np.float32)
            _, result = classifier.predict(test_x)
            predict_y = result[0][0]

            if predict_y:
                x1, y1 = tuple((joint.min(axis=0)[:2] * [frame.shape[1], frame.shape[0]] * 0.85).astype(int))
                x2, y2 = tuple((joint.max(axis=0)[:2] * [frame.shape[1], frame.shape[0]] * 1.1).astype(int))

                frame_mosaic = frame[y1:y2, x1:x2].copy()
                frame_mosaic = cv2.resize(frame_mosaic, dsize=None, fx=0.05, fy=0.05, interpolation=cv2.INTER_NEAREST)
                frame_mosaic = cv2.resize(frame_mosaic, dsize=(x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

                y1 = 0 if y1 < 0 else y1
                y2 = 479 if y2 > 479 else y2
                x1 = 0 if x1 < 0 else x1
                x2 = 639 if x2 > 639 else x2
                frame[y1:y2, x1:x2] = frame_mosaic

    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()