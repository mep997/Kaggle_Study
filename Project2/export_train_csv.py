import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

hands = mp.solutions.hands.Hands()

recording_data = []
recording = False
gesture = False
while True:
    keycode = cv2.waitKey(30)
    if ord('q') == keycode:
        break
    elif ord('r') == keycode:
        recording = not recording
    elif ord('g') == keycode:
        gesture = not gesture

    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if recording:
        text = '{}'.format('True' if gesture else 'False')
        frame  = cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, COLOR_RED)
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
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

            angle = np.degrees(angle)
            recording_data.append(angle.tolist() + [1 if gesture else 0])

    cv2.imshow("Camera", frame)

if recording_data:
    df = pd.DataFrame(recording_data)
    df.to_csv('custom_train.csv', index=False)

capture.release()
cv2.destroyAllWindows()