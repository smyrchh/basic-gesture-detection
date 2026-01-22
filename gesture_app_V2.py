import cv2
import mediapipe as mp
import numpy as np
import os

# Load meme images
def load_images(folder):
    names = {
        "smile": "smile.jpeg",
        "shock": "shock-disgusting.jpeg", 
        "thumbs_up": "thumbs-up.jpeg",
        "thumbs_down": "thumbs-down.jpeg",
        "tongue_out": "tongue-out.jpeg"
    }
    images = {}
    for key, filename in names.items():
        path = os.path.join(folder, "jpeg", filename)
        img = cv2.imread(path)
        if img is not None:
            images[key] = img
        else:
            print(f"Warning: Could not load {path}")
    return images

# Simple smile detector
def mouth_open(landmarks):
    return abs(landmarks[14].y - landmarks[13].y) > 0.03

def smile_ratio(landmarks):
    left, right = 61, 291
    cheeks = 234, 454
    mouth = np.hypot(landmarks[right].x - landmarks[left].x,
                     landmarks[right].y - landmarks[left].y)
    face = np.hypot(landmarks[cheeks[1]].x - landmarks[cheeks[0]].x,
                    landmarks[cheeks[1]].y - landmarks[cheeks[0]].y)
    return mouth / face if face != 0 else 0

def smiling(landmarks):
    return smile_ratio(landmarks) > 0.4

# Thumb gestures
def thumb_direction(hand):
    wrist = hand[0].y
    thumb = hand[4].y
    if thumb < wrist - 0.05:
        return "thumbs_up"
    elif thumb > wrist + 0.05:
        return "thumbs_down"
    return None

def main():
    mp_face = mp.solutions.face_mesh.FaceMesh()
    mp_hands = mp.solutions.hands.Hands()
    cap = cv2.VideoCapture(0)
    base = os.path.dirname(__file__)
    imgs = load_images(base)

    print("Press Q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face = mp_face.process(rgb)
        hands = mp_hands.process(rgb)

        gesture = "shock"

        if face.multi_face_landmarks:
            lm = face.multi_face_landmarks[0].landmark
            if mouth_open(lm):
                gesture = "tongue_out"
            elif smiling(lm):
                gesture = "smile"

        if hands.multi_hand_landmarks:
            for h in hands.multi_hand_landmarks:
                result = thumb_direction(h.landmark)
                if result:
                    gesture = result

        meme = cv2.resize(imgs[gesture], (640, 480))
        cv2.imshow("Meme", meme)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
