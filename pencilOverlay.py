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
    # Initialize servos at neutral (90° ~ 7.5% duty cycle)
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

# --- Computer Vision: Pencil Detection & Overlay ---

def process_frame(frame):
    """
    Processes the frame to detect the pencil and compute three errors:
      - pitch_error: vertical offset from the image center.
      - yaw_error: horizontal offset from the image center.
      - roll_error: difference between the pencil's line angle and vertical (90°).
    It overlays the detection information on the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    # Assume the largest contour is the pencil
    pencil_contour = max(contours, key=cv2.contourArea)
    
    # Draw the detected contour on the frame (blue)
    cv2.drawContours(frame, [pencil_contour], -1, (255, 0, 0), 2)
    
    # Compute centroid of the pencil contour
    M = cv2.moments(pencil_contour)
    if M['m00'] != 0:
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
    else:
        cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
    
    # Fit a line to the contour points and draw it (green)
    [vx, vy, x, y] = cv2.fitLine(pencil_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    rows, cols = frame.shape[:2]
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(frame, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
    
    # Calculate the pencil's angle in degrees
    angle = np.arctan2(vy, vx) * (180 / np.pi)
    # For roll correction, assume the pencil should be vertical (90°)
    roll_error = 90 - abs(angle)
    
    # Determine errors based on the centroid offset from the image center
    frame_center_x = frame.shape[1] / 2
    frame_center_y = frame.shape[0] / 2
    yaw_error = frame_center_x - cX    # positive: pencil is left of center
    pitch_error = frame_center_y - cY    # positive: pencil is above center

    # Overlay error and angle information on the frame (yellow text)
    overlay_text = f"Angle: {angle[0]:.2f} deg, Roll Err: {roll_error:.2f}, Yaw Err: {yaw_error:.2f}, Pitch Err: {pitch_error:.2f}"
    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return pitch_error, yaw_error, roll_error

# --- Main Loop: Capture, Process, and Control ---

def main():
    # Initialize servos (even if not physically connected)
    yaw_pwm, pitch_pwm, roll_pwm = initialize_servos()

    # Use a GStreamer pipeline with libcamerasrc and sync=false
    pipeline = (
        "libcamerasrc ! video/x-raw,width=320,height=240,framerate=30/1 ! "
        "videoconvert ! appsink sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device. Ensure libcamera is installed and enabled.")
        return

    time.sleep(2)  # Allow camera sensor to warm up
    print("Camera stream opened successfully.")

    # Gain factors for testing overlay (even if servos aren't connected)
    pitch_gain = 0.05
    yaw_gain = 0.05
    roll_gain = 0.1

    current_pitch = 90
    current_yaw = 90
    current_roll = 90

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break

            # Debug: Print frame dimensions
            print(f"Captured frame with shape: {frame.shape}")

            pitch_error, yaw_error, roll_error = process_frame(frame)
            if pitch_error is not None:
                new_pitch = current_pitch + pitch_gain * pitch_error
                new_yaw = current_yaw + yaw_gain * yaw_error
                new_roll = current_roll + roll_gain * roll_error

                new_pitch = max(0, min(180, new_pitch))
                new_yaw = max(0, min(180, new_yaw))
                new_roll = max(0, min(180, new_roll))

                print(f"Pitch Err: {pitch_error:.2f}, Yaw Err: {yaw_error:.2f}, Roll Err: {roll_error:.2f}")
                print(f"Virtual Servo Angles -> Pitch: {new_pitch:.2f}, Yaw: {new_yaw:.2f}, Roll: {new_roll:.2f}")

                current_pitch, current_yaw, current_roll = new_pitch, new_yaw, new_roll

            cv2.imshow("Pencil Orientation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        yaw_pwm.stop()
        pitch_pwm.stop()
        roll_pwm.stop()
        GPIO.cleanup()
        del yaw_pwm, pitch_pwm, roll_pwm

if __name__ == '__main__':
    main()
