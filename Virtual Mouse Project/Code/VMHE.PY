import cv2
import mediapipe as mp
import pyautogui

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.flip(image, 1)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(image, bbox, (255, 0, 255), 2)
                landmarks = detection.location_data.relative_keypoints
                for lm in landmarks:
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(image, (x, y), 5, (0, 255, 255), -1)

                if len(landmarks) >= 2:
                    left_eye_x = int(landmarks[0].x * iw)
                    left_eye_y = int(landmarks[0].y * ih)
                    right_eye_x = int(landmarks[1].x * iw)
                    right_eye_y = int(landmarks[1].y * ih)
                    eye_x = (left_eye_x + right_eye_x) // 2
                    eye_y = (left_eye_y + right_eye_y) // 2
                    mouse_x = int(eye_x * screen_width / iw)
                    mouse_y = int(eye_y * screen_height / ih)
                    pyautogui.moveTo(mouse_x, mouse_y)

        cv2.imshow('Virtual Mouse', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
