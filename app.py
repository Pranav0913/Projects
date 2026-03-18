import cv2
import mediapipe as mp
import pyautogui
import math

# Webcam setup
cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_w, screen_h = pyautogui.size()

# Smooth movement variables
prev_x, prev_y = 0, 0
smoothening = 5

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    h, w, c = img.shape

    # Convert to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_img)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []

            # Get landmark positions
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # Index finger tip (8) & Thumb tip (4)
            x1, y1 = lm_list[8]
            x2, y2 = lm_list[4]

            # Convert coordinates
            screen_x = screen_w * x1 / w
            screen_y = screen_h * y1 / h

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            # Click detection (pinch)
            distance = math.hypot(x2 - x1, y2 - y1)

            if distance < 30:
                pyautogui.click()
                pyautogui.sleep(1)

            # Draw landmarks
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Show window
    cv2.imshow("Virtual Mouse", img)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()