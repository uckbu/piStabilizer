import cv2
import numpy as np
import time
from picamera2 import Picamera2
import paho.mqtt.client as mqtt

# MQTT configuration – change these to match your network/broker
BROKER_ADDRESS = "192.168.1.208"  # Replace with your MQTT broker IP
BROKER_PORT = 1883
MQTT_TOPIC = "servo/angles"

# Initialize MQTT client
mqtt_client = mqtt.Client("PiServoPublisher")
mqtt_client.connect(BROKER_ADDRESS, BROKER_PORT, 60)
time.sleep(1)  # Give a moment for the connection to establish

# --- Virtual Servo Control Simulation ---
def set_virtual_servo_angle(current_angle, error, gain):
    new_angle = current_angle + gain * error
    return max(0, min(180, new_angle))

# --- Computer Vision: Yellow Pencil Tracking & Overlay ---
def process_frame(frame):
    """
    Detects the largest object in the yellow range (assumed to be a #2 Dixon Ticonderoga pencil)
    and computes errors:
      - pitch_error: vertical offset from the image center.
      - yaw_error: horizontal offset from the image center.
      - roll_error: deviation of the pencil's orientation from vertical.
    Overlays detection info on the frame.
    """
    # Convert from RGB (Picamera2 default) to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    # Define HSV range for a typical bright #2 pencil yellow.
    # These values may need further tuning based on your lighting conditions.
    lower_bound = np.array([5, 108, 96])
    upper_bound = np.array([40, 255, 240])
    
    # Create mask to extract yellow regions
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    # Filter contours by area and aspect ratio (pencil-like objects are elongated)
    MIN_CONTOUR_AREA = 500
    MIN_ASPECT_RATIO = 3.0  # Adjust based on your pencil dimensions

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

    # Select the largest valid contour (assumed to be the pencil)
    pencil_contour = max(valid_contours, key=cv2.contourArea)
    cv2.drawContours(frame, [pencil_contour], -1, (255, 0, 0), 2)
    
    # Compute the centroid of the pencil contour
    M = cv2.moments(pencil_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = frame.shape[1] // 2, frame.shape[0] // 2
    cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
    
    # Fit a line to the contour points to estimate orientation
    line = cv2.fitLine(pencil_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = map(float, line.squeeze())
    rows, cols = frame.shape[:2]
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(frame, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
    
    # Calculate the pencil's angle and compute roll error (difference from vertical)
    angle = float(np.arctan2(vy, vx) * (180 / np.pi))
    roll_error = 90 - abs(angle)
    
    # Calculate error relative to frame center
    frame_center_x = frame.shape[1] / 2
    frame_center_y = frame.shape[0] / 2
    yaw_error = frame_center_x - cX
    pitch_error = frame_center_y - cY

    overlay_text = (f"Angle: {angle:.2f}°, Roll Err: {roll_error:.2f}, "
                    f"Yaw Err: {yaw_error:.2f}, Pitch Err: {pitch_error:.2f}")
    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2)

    return pitch_error, yaw_error, roll_error

# --- Main Loop: Capture, Process, Publish, and Simulate Control ---
def main():
    picam2 = Picamera2()
    video_config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(video_config)
    picam2.start()
    
    # Give the camera time to warm up
    time.sleep(2)
    print("Camera stream initialized successfully via Picamera2.")

    # Gain factors for virtual servo simulation
    pitch_gain = 0.05
    yaw_gain = 0.05
    roll_gain = 0.1

    # Starting angles for virtual servos
    current_pitch = 90
    current_yaw   = 90
    current_roll  = 90

    try:
        while True:
            # Capture frame from Picamera2 (in RGB)
            frame = picam2.capture_array()
            if frame is None:
                print("Failed to capture frame.")
                break

            pitch_error, yaw_error, roll_error = process_frame(frame)
            if pitch_error is not None:
                current_pitch = set_virtual_servo_angle(current_pitch, pitch_error, pitch_gain)
                current_yaw   = set_virtual_servo_angle(current_yaw, yaw_error, yaw_gain)
                current_roll  = set_virtual_servo_angle(current_roll, roll_error, roll_gain)

                print(f"Pitch Err: {pitch_error:.2f}, Yaw Err: {yaw_error:.2f}, Roll Err: {roll_error:.2f}")
                print(f"Virtual Servo Angles -> Pitch: {current_pitch:.2f}, Yaw: {current_yaw:.2f}, Roll: {current_roll:.2f}")

                # Publish the servo angles over MQTT
                message = f"{current_pitch},{current_yaw},{current_roll}\n"
                mqtt_client.publish(MQTT_TOPIC, message)
                print("Published:", message.strip())

            cv2.imshow("Yellow Pencil Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
