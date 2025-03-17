import cv2
import numpy as np
import time
import RPi.GPIO as GPIO

# --- Configuration ---
# GPIO pins for each servo (using BCM numbering)
SERVO_PITCH_PIN = 17
SERVO_YAW_PIN   = 27
SERVO_ROLL_PIN  = 22

PWM_FREQUENCY = 50  # 50 Hz for SG90 servos

# --- Helper Functions ---
def angle_to_duty(angle):
    """
    Convert an angle in degrees (0-180) to a duty cycle percentage.
    For SG90, 0° typically corresponds to ~2.5% duty cycle and 180° to ~12.5%.
    """
    return 2.5 + (angle / 180.0) * 10.0

def set_servo_angle(servo_pwm, angle):
    """
    Update servo position by changing the PWM duty cycle.
    """
    duty = angle_to_duty(angle)
    servo_pwm.ChangeDutyCycle(duty)

# --- Computer Vision: Pencil Tracking ---
def process_frame(frame):
    """
    Detects the largest yellow-orange object (assumed to be a pencil) and
    computes errors relative to the frame center:
      - pitch_error: vertical offset.
      - yaw_error: horizontal offset.
      - roll_error: difference between the pencil’s angle and vertical.
    Overlays detection information on the frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([10, 100, 100])
    upper_bound = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Reduce noise with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    MIN_CONTOUR_AREA = 500
    MIN_ASPECT_RATIO = 3.0
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue
        aspect_ratio = float(max(w, h) / min(w, h))
        if aspect_ratio < MIN_ASPECT_RATIO:
            continue
        valid_contours.append(cnt)
    
    if not valid_contours:
        return None, None, None

    pencil_contour = max(valid_contours, key=cv2.contourArea)
    cv2.drawContours(frame, [pencil_contour], -1, (255, 0, 0), 2)
    
    M = cv2.moments(pencil_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
    
    # Fit a line to estimate the pencil's orientation
    line = cv2.fitLine(pencil_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = map(float, line.squeeze())
    rows, cols = frame.shape[:2]
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(frame, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
    
    angle = float(np.arctan2(vy, vx) * (180 / np.pi))
    roll_error = 90 - abs(angle)
    
    frame_center_x = frame.shape[1] / 2
    frame_center_y = frame.shape[0] / 2
    yaw_error = frame_center_x - cX
    pitch_error = frame_center_y - cY

    overlay_text = (f"Angle: {angle:.2f}°  Roll Err: {roll_error:.2f}  "
                    f"Yaw Err: {yaw_error:.2f}  Pitch Err: {pitch_error:.2f}")
    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return pitch_error, yaw_error, roll_error

# --- Main Loop ---
def main():
    # --- GPIO Setup ---
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PITCH_PIN, GPIO.OUT)
    GPIO.setup(SERVO_YAW_PIN, GPIO.OUT)
    GPIO.setup(SERVO_ROLL_PIN, GPIO.OUT)
    
    servo_pitch = GPIO.PWM(SERVO_PITCH_PIN, PWM_FREQUENCY)
    servo_yaw   = GPIO.PWM(SERVO_YAW_PIN, PWM_FREQUENCY)
    servo_roll  = GPIO.PWM(SERVO_ROLL_PIN, PWM_FREQUENCY)
    
    # Start PWM with initial duty cycle corresponding to 90° (neutral)
    initial_angle = 90
    servo_pitch.start(angle_to_duty(initial_angle))
    servo_yaw.start(angle_to_duty(initial_angle))
    servo_roll.start(angle_to_duty(initial_angle))
    
    # Setup camera capture (for Pi Camera Module 3, ensure libcamera is enabled if needed)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        GPIO.cleanup()
        return

    time.sleep(2)
    print("Camera stream opened successfully.")

    # Gain factors for servo adjustments based on computed error values
    pitch_gain = 0.05
    yaw_gain = 0.05
    roll_gain = 0.1

    current_pitch = initial_angle
    current_yaw   = initial_angle
    current_roll  = initial_angle

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            pitch_error, yaw_error, roll_error = process_frame(frame)
            if pitch_error is not None:
                # Update servo angles based on error corrections
                current_pitch = max(0, min(180, current_pitch + pitch_gain * pitch_error))
                current_yaw   = max(0, min(180, current_yaw   + yaw_gain * yaw_error))
                current_roll  = max(0, min(180, current_roll  + roll_gain * roll_error))

                print(f"Pitch Err: {pitch_error:.2f}, Yaw Err: {yaw_error:.2f}, Roll Err: {roll_error:.2f}")
                print(f"Servo Angles -> Pitch: {current_pitch:.2f}, Yaw: {current_yaw:.2f}, Roll: {current_roll:.2f}")

                # Move the servos to the new angles by updating PWM duty cycles
                set_servo_angle(servo_pitch, current_pitch)
                set_servo_angle(servo_yaw, current_yaw)
                set_servo_angle(servo_roll, current_roll)

            cv2.imshow("Yellow-Orange Pencil Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        servo_pitch.stop()
        servo_yaw.stop()
        servo_roll.stop()
        GPIO.cleanup()

if __name__ == '__main__':
    main()
