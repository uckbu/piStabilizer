import cv2
import numpy as np
import time
from picamera2 import Picamera2

def nothing(x):
    """Callback for trackbar (does nothing)."""
    pass

def main():
    # --- 1) Initialize the camera (Picamera2) ---
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()

    time.sleep(2)
    print("Picamera2 started. Adjust the trackbars to find your pencil's HSV range.")

    # --- 2) Create a window and trackbars to adjust HSV bounds ---
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trackbars", 400, 300)

    # Initial guesses for a bright #2 pencil (tweak as needed)
    cv2.createTrackbar("LH", "Trackbars", 20, 179, nothing)  # Lower Hue
    cv2.createTrackbar("LS", "Trackbars", 100, 255, nothing) # Lower Saturation
    cv2.createTrackbar("LV", "Trackbars", 100, 255, nothing) # Lower Value

    cv2.createTrackbar("UH", "Trackbars", 40, 179, nothing)  # Upper Hue
    cv2.createTrackbar("US", "Trackbars", 255, 255, nothing) # Upper Saturation
    cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing) # Upper Value

    try:
        while True:
            frame = picam2.capture_array()
            if frame is None:
                print("Failed to capture frame.")
                break

            # --- 3) Read current trackbar positions (HSV bounds) ---
            lh = cv2.getTrackbarPos("LH", "Trackbars")
            ls = cv2.getTrackbarPos("LS", "Trackbars")
            lv = cv2.getTrackbarPos("LV", "Trackbars")

            uh = cv2.getTrackbarPos("UH", "Trackbars")
            us = cv2.getTrackbarPos("US", "Trackbars")
            uv = cv2.getTrackbarPos("UV", "Trackbars")

            lower_bound = np.array([lh, ls, lv])
            upper_bound = np.array([uh, us, uv])

            # Convert RGB (Picamera2 default) to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            # --- 4) Create a mask for the current HSV range ---
            mask = cv2.inRange(hsv, lower_bound, upper_bound)

            # Optionally apply some morphological ops to clean noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # --- 5) Show the live camera feed and the mask side by side ---
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack((frame, mask_bgr))

            cv2.imshow("Camera (left) + Mask (right)", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
