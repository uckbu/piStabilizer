import cv2
import mediapipe as mp
import numpy as np
import time
import RPi.GPIO as GPIO
from picamera2 import Picamera2  # Import the Picamera2 library

# ---------- Servo Setup ----------
GPIO.setmode(GPIO.BCM)
# Define GPIO pins for SG90 servos
PITCH_PIN = 17
YAW_PIN   = 27
ROLL_PIN  = 22

GPIO.setup(PITCH_PIN, GPIO.OUT)
GPIO.setup(YAW_PIN, GPIO.OUT)
GPIO.setup(ROLL_PIN, GPIO.OUT)

# Initialize PWM for each servo at 50Hz
pitch_servo = GPIO.PWM(PITCH_PIN, 50)
yaw_servo   = GPIO.PWM(YAW_PIN, 50)
roll_servo  = GPIO.PWM(ROLL_PIN, 50)

pitch_servo.start(0)
yaw_servo.start(0)
roll_servo.start(0)

def update_servo(servo, angle):
    # Map angle (0-180) to duty cycle (approx. 2.5% - 12.5%)
    duty = (angle / 18.0) + 2.5
    servo.ChangeDutyCycle(duty)

# ---------- Helper Functions ----------

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def exponential_smoothing(new_val, old_val, alpha=0.3):
    """Smooth new_val with the previous value using factor alpha."""
    return alpha * new_val + (1 - alpha) * old_val

def set_virtual_servo_angle(current_angle, error, gain):
    """Simulate servo update by adding gain*error; clamp result to [0,180]."""
    new_angle = current_angle + gain * error
    return max(0, min(180, new_angle))

def compute_euler_angles(hand_landmarks, w, h):
    """
    Compute a 3D orientation from hand landmarks:
      - Pencil axis: vector from THUMB_TIP to INDEX_FINGER_TIP.
      - Pitch, yaw from that axis (camera coords: x right, y down, z forward).
      - Returns a fallback roll from the same axis, if needed.
    """
    mp_h = mp.solutions.hands.HandLandmark

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
    
    pencil_dir = normalize(index_tip - thumb_tip)
    
    yaw = np.degrees(np.arctan2(pencil_dir[0], pencil_dir[2]))
    pitch = np.degrees(np.arctan2(-pencil_dir[1],
                                  np.sqrt(pencil_dir[0]**2 + pencil_dir[2]**2)))
    
    dx = index_tip[0] - thumb_tip[0]
    dy = index_tip[1] - thumb_tip[1]
    fallback_roll = np.degrees(np.arctan2(dy, dx))
    
    return pitch, yaw, fallback_roll, pencil_dir

def compute_lavender_pen_roll(frame):
    """
    Detects a lavender pen by color segmentation (approx. HSV for pastel purple),
    extracts its contour, and fits a line to compute orientation (roll).
    
    Returns:
      measured_roll: angle (in degrees) of the pen's axis,
                     near 90° if the pen is upright.
      None if no lavender pen is detected.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_lavender = np.array([130, 50, 70])
    upper_lavender = np.array([170, 255, 255])
    mask = cv2.inRange(hsv, lower_lavender, upper_lavender)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    MIN_AREA = 300
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    if not valid_contours:
        return None
    
    pen_contour = max(valid_contours, key=cv2.contourArea)
    
    cv2.drawContours(frame, [pen_contour], -1, (255, 0, 255), 2)
    
    line = cv2.fitLine(pen_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = map(float, line.squeeze())
    measured_angle = np.degrees(np.arctan2(vy, vx))
    return measured_angle

# ---------- Main Application ----------

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    # Initialize Picamera2 and configure the camera for BGR output at 640x480
    picam2 = Picamera2()
    video_config = picam2.create_preview_configuration(
        main={"format": "BGR888", "size": (640, 480)}
    )
    picam2.configure(video_config)
    picam2.start()
    
    # Give the camera time to warm up
    time.sleep(2)
    print("Camera initialized successfully via Picamera2.")
    
    # Obtain frame dimensions from a captured frame
    frame = picam2.capture_array()
    if frame is None:
        print("Error: Could not read frame from camera.")
        picam2.stop()
        return
    h, w = frame.shape[:2]
    
    smoothed = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    alpha = 0.3

    ref_angles = None
    current_pitch = 90
    current_yaw   = 90
    current_roll  = 90
    
    pitch_gain = 0.05
    yaw_gain   = 0.05
    roll_gain  = 0.1
    
    print("Press 'c' to calibrate (set current orientation as neutral). Press 'q' to quit.")
    
    try:
        while True:
            # Capture frame from Picamera2
            frame = picam2.capture_array()
            if frame is None:
                print("Failed to capture frame.")
                break
            
            # Process the image for hand landmarks using MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            pitch, yaw, fallback_roll, _ = 0, 0, 0, None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                pitch, yaw, fallback_roll, _ = compute_euler_angles(hand_landmarks, w, h)
            
            pen_roll = compute_lavender_pen_roll(frame)
            
            if pen_roll is not None:
                raw_roll = pen_roll
            else:
                raw_roll = fallback_roll
            
            smoothed["pitch"] = exponential_smoothing(pitch, smoothed["pitch"], alpha)
            smoothed["yaw"]   = exponential_smoothing(yaw,   smoothed["yaw"],   alpha)
            smoothed["roll"]  = exponential_smoothing(raw_roll, smoothed["roll"], alpha)
            
            if ref_angles is not None:
                error_pitch = smoothed["pitch"] - ref_angles[0]
                error_yaw   = smoothed["yaw"]   - ref_angles[1]
                error_roll  = smoothed["roll"]  - ref_angles[2]
            else:
                error_pitch = smoothed["pitch"]
                error_yaw   = smoothed["yaw"]
                error_roll  = smoothed["roll"]
            
            current_pitch = set_virtual_servo_angle(current_pitch, error_pitch, pitch_gain)
            current_yaw   = set_virtual_servo_angle(current_yaw,   error_yaw,   yaw_gain)
            current_roll  = set_virtual_servo_angle(current_roll,  error_roll,  roll_gain)
            
            update_servo(pitch_servo, current_pitch)
            update_servo(yaw_servo, current_yaw)
            update_servo(roll_servo, current_roll)
            
            font_scale = 0.6
            color = (0, 255, 0)
            thickness = 2
            cv2.putText(frame, f"Pitch: {current_pitch:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.putText(frame, f"Yaw:   {current_yaw:.2f}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.putText(frame, f"Roll:  {current_roll:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            
            if ref_angles is not None:
                cv2.putText(frame, "(Calibrated)", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            debug_text = f"Raw P:{pitch:.1f} Y:{yaw:.1f} R:{raw_roll:.1f}"
            cv2.putText(frame, debug_text, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Hand & Lavender Pen Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                ref_angles = (smoothed["pitch"], smoothed["yaw"], smoothed["roll"])
                print(f"Calibrated: Pitch {ref_angles[0]:.2f}, Yaw {ref_angles[1]:.2f}, Roll {ref_angles[2]:.2f}")
    finally:
        pitch_servo.stop()
        yaw_servo.stop()
        roll_servo.stop()
        GPIO.cleanup()
        picam2.stop()  # Stop Picamera2
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
