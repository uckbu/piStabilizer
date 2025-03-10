import cv2
import numpy as np
import time
import RPi.GPIO as GPIO

# --- Hardware Setup ---

# Define GPIO pins for the servos (update with your wiring)
YAW_PIN = 17
PITCH_PIN = 27
ROLL_PIN = 22

def initialize_servos():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(YAW_PIN, GPIO.OUT)
    GPIO.setup(PITCH_PIN, GPIO.OUT)
    GPIO.setup(ROLL_PIN, GPIO.OUT)
    # Set up PWM for each servo at 50Hz (typical for hobby servos)
    yaw_pwm = GPIO.PWM(YAW_PIN, 50)
    pitch_pwm = GPIO.PWM(PITCH_PIN, 50)
    roll_pwm = GPIO.PWM(ROLL_PIN, 50)
    # Initialize servos at neutral (typically 90° corresponding to ~7.5% duty cycle)
    yaw_pwm.start(7.5)
    pitch_pwm.start(7.5)
    roll_pwm.start(7.5)
    return yaw_pwm, pitch_pwm, roll_pwm

def set_servo_angle(pwm, angle):
    """
    Map an angle (0-180°) to the corresponding duty cycle.
    Adjust the mapping if your servos require different calibration.
    """
    duty = 2.5 + (angle / 180.0) * 10.0
    pwm.ChangeDutyCycle(duty)

# --- Computer Vision: Pencil Orientation Detection ---

def process_frame(frame):
    """
    Processes the frame to detect the pencil and compute three errors:
      - pitch_error: vertical offset from the image center.
      - yaw_error: horizontal offset from the image center.
      - roll_error: difference between the pencil's line angle and vertical (90°).
    Returns a tuple (pitch_error, yaw_error, roll_error).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    # Assume the largest contour is the pencil
    pencil_contour = max(contours, key=cv2.contourArea)
    
    # Compute centroid of the pencil contour
    M = cv2.moments(pencil_contour)
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    else:
        cX, cY = frame.shape[1] // 2, frame.shape[0] // 2

    # Fit a line to the contour points
    [vx, vy, x, y] = cv2.fitLine(pencil_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.arctan2(vy, vx) * (180 / np.pi)
    # For roll correction, we assume the pencil should be vertical (90°).
    roll_error = 90 - abs(angle)
    
    # Determine errors based on the centroid offset from the image center
    frame_center_x = frame.shape[1] / 2
    frame_center_y = frame.shape[0] / 2
    yaw_error = frame_center_x - cX    # positive error: pencil is left of center
    pitch_error = frame_center_y - cY  # positive error: pencil is above center
    
    return pitch_error, yaw_error, roll_error

# --- Main Loop: Capture, Process, and Control ---

def main():
    # Initialize servos and camera
    yaw_pwm, pitch_pwm, roll_pwm = initialize_servos()
    cap = cv2.VideoCapture(0)  # Adjust if your camera index is different
    time.sleep(2)  # Allow camera sensor to warm up

    # Gain factors for each control axis (tune these experimentally)
    pitch_gain = 0.05  # Conversion factor: degrees per pixel error
    yaw_gain = 0.05
    roll_gain = 0.1   # Roll error is already in degrees

    # Start with neutral positions (90°)
    current_pitch = 90
    current_yaw = 90
    current_roll = 90

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            pitch_error, yaw_error, roll_error = process_frame(frame)
            if pitch_error is not None:
                # Calculate new servo angles based on errors and gains
                new_pitch = current_pitch + pitch_gain * pitch_error
                new_yaw = current_yaw + yaw_gain * yaw_error
                new_roll = current_roll + roll_gain * roll_error

                # Constrain the angles to the valid range (0-180°)
                new_pitch = max(0, min(180, new_pitch))
                new_yaw = max(0, min(180, new_yaw))
                new_roll = max(0, min(180, new_roll))

                # Update each servo
                set_servo_angle(pitch_pwm, new_pitch)
                set_servo_angle(yaw_pwm, new_yaw)
                set_servo_angle(roll_pwm, new_roll)

                print(f"Pitch error: {pitch_error:.2f}, Yaw error: {yaw_error:.2f}, Roll error: {roll_error:.2f}")
                print(f"Servo angles -> Pitch: {new_pitch:.2f}, Yaw: {new_yaw:.2f}, Roll: {new_roll:.2f}")

                # Update current positions (could be smoothed for better performance)
                current_pitch = new_pitch
                current_yaw = new_yaw
                current_roll = new_roll
            
            # Optional: display the frame for debugging
            cv2.imshow("Pencil Orientation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

if __name__ == '__main__':
    main()
