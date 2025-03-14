import cv2
import mediapipe as mp
import numpy as np
import time
import pigpio  # Library for PWM control on Raspberry Pi

# ---------- Servo Configuration ----------
SERVO_PINS = {"pitch": 17, "yaw": 27, "roll": 22}  # GPIO pins for servos
SERVO_RANGE = (500, 2500)  # Servo pulse range (in microseconds)
SERVO_NEUTRAL = 1500  # Neutral position

pi = pigpio.pi()  # Initialize pigpio
if not pi.connected:
    print("Error: Unable to connect to pigpio daemon.")
    exit()

# Set up servos
for pin in SERVO_PINS.values():
    pi.set_servo_pulsewidth(pin, SERVO_NEUTRAL)

# ---------- Helper Functions ----------

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def exponential_smoothing(new_val, old_val, alpha=0.3):
    """Smooth new_val with the previous value using factor alpha."""
    return alpha * new_val + (1 - alpha) * old_val

def set_servo_angle(servo, angle):
    """Map angle (0-180) to servo pulse width (500-2500us)."""
    pulse_width = np.interp(angle, [0, 180], SERVO_RANGE)
    pi.set_servo_pulsewidth(SERVO_PINS[servo], pulse_width)

def compute_euler_angles(hand_landmarks):
    mp_h = mp.solutions.hands.HandLandmark

    # Get thumb and index fingertips in normalized coordinates
    thumb_tip = np.array([
        hand_landmarks.landmark[mp_h.THUMB_TIP].x,
        hand_landmarks.landmark[mp_h.THUMB_TIP].y,
        hand_landmarks.landmark[mp_h.THUMB_TIP].z
    ])
    index_tip = np.array([
        hand_landmarks.landmark[mp_h.INDEX_FINGER_TIP].x,
        hand_landmarks.landmark[mp_h.INDEX_FINGER_TIP].y,
        hand_landmarks.landmark[mp_h.INDEX_FINGER_TIP].z
    ])
    
    # Pencil axis vector
    pencil_dir = normalize(index_tip - thumb_tip)
    
    yaw = np.degrees(np.arctan2(pencil_dir[0], pencil_dir[2]))
    pitch = np.degrees(np.arctan2(-pencil_dir[1], np.sqrt(pencil_dir[0]**2 + pencil_dir[2]**2)))
    roll = np.degrees(np.arctan2(index_tip[1] - thumb_tip[1], index_tip[0] - thumb_tip[0]))
    
    return pitch, yaw, roll

# ---------- Main Application ----------

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    time.sleep(2)
    print("Webcam stream opened successfully.")

    # Smoothing variables
    smoothed = {"pitch": 90.0, "yaw": 90.0, "roll": 90.0}
    alpha = 0.3  # smoothing factor

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                pitch, yaw, roll = compute_euler_angles(hand_landmarks)
                
                # Exponential smoothing
                smoothed["pitch"] = exponential_smoothing(pitch, smoothed["pitch"], alpha)
                smoothed["yaw"] = exponential_smoothing(yaw, smoothed["yaw"], alpha)
                smoothed["roll"] = exponential_smoothing(roll, smoothed["roll"], alpha)

                # Convert to servo angles and update servos
                set_servo_angle("pitch", smoothed["pitch"])
                set_servo_angle("yaw", smoothed["yaw"])
                set_servo_angle("roll", smoothed["roll"])

            # Display debug info
            cv2.putText(frame, f"Pitch: {smoothed['pitch']:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {smoothed['yaw']:.2f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {smoothed['roll']:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Hand Tracking Gimbal", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        for pin in SERVO_PINS.values():
            pi.set_servo_pulsewidth(pin, 0)  # Turn off servos
        pi.stop()

if __name__ == "__main__":
    main()
