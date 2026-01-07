import cv2
import mediapipe as mp
import numpy as np
import json
import threading
import serial
import time
import os
import pyvirtualcam



arduino_distance = -1

def read_arduino():
    global arduino_distance
    
    while True:
        try:
            print("Attempting COM3 connection...")
            ser = serial.Serial('COM3', 115200, timeout=1)
            print("✔ Arduino connected on COM3.")
            
            while True:
                try:
                    line = ser.readline().decode().strip()
                    if line:
                        data = json.loads(line)
                        arduino_distance = data["distance"]
                except:
                    pass

        except Exception as e:
            print("❌ Arduino COM3 error:", e)
            print("Retrying in 2 seconds...")
            time.sleep(2)

threading.Thread(target=read_arduino, daemon=True).start()


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=5,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FPS, 30)


def classify_gaze(iris_x, left_x, right_x):
    ratio = (iris_x - left_x) / (right_x - left_x + 1e-6)
    if ratio < 0.33:
        return "LEFT"
    elif ratio < 0.66:
        return "CENTER"
    else:
        return "RIGHT"


with pyvirtualcam.Camera(width=960, height=540, fps=30) as cam:
    print(f"✔ Virtual camera running: {cam.device}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        faces_packet = []
        face_id = 0

        # Only show visuals if distance <= 60cm
        show_visuals = (arduino_distance != -1 and arduino_distance <= 60)

        # Clean output frame for OBS VirtualCam
        output_frame = frame.copy()


        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                LEFT_IRIS = 468
                LEFT_LEFT = 33
                LEFT_RIGHT = 133

                iris = face_landmarks.landmark[LEFT_IRIS]
                left_corner = face_landmarks.landmark[LEFT_LEFT]
                right_corner = face_landmarks.landmark[LEFT_RIGHT]

                gaze_zone = classify_gaze(iris.x, left_corner.x, right_corner.x)

                faces_packet.append({
                    "id": face_id,
                    "looking": gaze_zone
                })

                if show_visuals:
                    h, w, _ = output_frame.shape
                    iris_px = (int(iris.x * w), int(iris.y * h))

               
                    colour = (57,255,20) if gaze_zone == "CENTER" \
                        else (255,20,147) if gaze_zone == "LEFT" \
                        else (0,255,255)

                    box_size = 40
                    cv2.rectangle(
                        output_frame,
                        (iris_px[0] - box_size, iris_px[1] - box_size),
                        (iris_px[0] + box_size, iris_px[1] + box_size),
                        colour, 3
                    )

                    cv2.putText(output_frame, gaze_zone,
                                (iris_px[0] + 10, iris_px[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

                face_id += 1


        packet = {
            "faces": faces_packet,
            "distance_cm": arduino_distance,
            "timestamp": time.time()
        }

        with open("tracker.json", "w") as f:
            json.dump(packet, f, indent=2)


    
        cv2.putText(output_frame, f"{arduino_distance:.1f}cm",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,255), 2)

        if not show_visuals:
            cv2.putText(output_frame, "Move closer to activate tracking (<60cm)",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,0,255), 2)


        output_frame = cv2.resize(output_frame, (960, 540))

        cam.send(output_frame)
        cam.sleep_until_next_frame()

        cv2.imshow("Eye Tracker + Arduino + VirtualCam", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
