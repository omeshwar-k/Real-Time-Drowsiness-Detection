import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import winsound
import os

EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 10

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEEP_SOUND = os.path.join(BASE_DIR, "beep.wav")

# Load face detector and landmark model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye landmark indexes
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))


def eye_aspect_ratio(eye):
    """Compute Eye Aspect Ratio (EAR)"""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


cap = cv2.VideoCapture(0)
counter = 0
alarm_on = False
print("[INFO] Drowsiness detection started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

        leftEye = landmarks[LEFT_EYE]
        rightEye = landmarks[RIGHT_EYE]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # Eyes closed → play sound
        if ear < EAR_THRESHOLD:
            counter += 1

            if counter >= EAR_CONSEC_FRAMES and not alarm_on:
                alarm_on = True
                winsound.PlaySound(
                    BEEP_SOUND,
                    winsound.SND_FILENAME |
                    winsound.SND_LOOP |
                    winsound.SND_ASYNC
                )
        else:
            # Eyes open → stop sound
            counter = 0
            if alarm_on:
                alarm_on = False
                winsound.PlaySound(None, winsound.SND_PURGE)

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
