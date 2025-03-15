import cv2
import mediapipe as mp
import numpy as np
import time

# ---------- Helper Functions ----------

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def exponential_smoothing(new_val, old_val, alpha=0.3):
    """Smooth new_val with the previous value using factor alpha."""
    return alpha * new_val + (1 - alpha) * old_val

def set_virtual_servo_angle(current_angle, error, gain):
    """Simulate servo update; clamp to [0,180]."""
    new_angle = current_angle + gain * error
    return max(0, min(180, new_angle))

def compute_euler_angles(hand_landmarks, w, h):
    """
    Computes a 3D orientation from hand landmarks.
    
    Uses:
      - Pencil axis: from THUMB_TIP to INDEX_FINGER_TIP.
      - Palm normal: computed from (WRIST, INDEX_FINGER_MCP, PINKY_MCP).
    
    Returns:
      pitch, yaw, roll  (in degrees)
      where:
        - pitch and yaw are derived from the pencil axis direction,
        - roll is computed as the signed angle between the projection of the palm normal and
          the projection of global up onto the plane perpendicular to the pencil axis.
    """
    mp_h = mp.solutions.hands.HandLandmark

    # Convert relevant landmarks to 3D numpy arrays in normalized coordinates.
    thumb_tip = np.array([hand_landmarks.landmark[mp_h.THUMB_TIP].x,
                           hand_landmarks.landmark[mp_h.THUMB_TIP].y,
                           hand_landmarks.landmark[mp_h.THUMB_TIP].z])
    index_tip = np.array([hand_landmarks.landmark[mp_h.INDEX_FINGER_TIP].x,
                           hand_landmarks.landmark[mp_h.INDEX_FINGER_TIP].y,
                           hand_landmarks.landmark[mp_h.INDEX_FINGER_TIP].z])
    
    # For palm normal, use wrist, index_MCP, and pinky_MCP:
    wrist = np.array([hand_landmarks.landmark[mp_h.WRIST].x,
                      hand_landmarks.landmark[mp_h.WRIST].y,
                      hand_landmarks.landmark[mp_h.WRIST].z])
    index_mcp = np.array([hand_landmarks.landmark[mp_h.INDEX_FINGER_MCP].x,
                          hand_landmarks.landmark[mp_h.INDEX_FINGER_MCP].y,
                          hand_landmarks.landmark[mp_h.INDEX_FINGER_MCP].z])
    pinky_mcp = np.array([hand_landmarks.landmark[mp_h.PINKY_MCP].x,
                          hand_landmarks.landmark[mp_h.PINKY_MCP].y,
                          hand_landmarks.landmark[mp_h.PINKY_MCP].z])
    
    # Pencil axis vector (from thumb to index)
    pencil_dir = normalize(index_tip - thumb_tip)
    
    # Compute pitch and yaw from pencil_dir.
    # In camera coordinates: x right, y down, z forward.
    # We use a common conversion: 
    #   yaw = arctan2(x, z), pitch = arctan2(-y, sqrt(x^2+z^2))
    yaw = np.degrees(np.arctan2(pencil_dir[0], pencil_dir[2]))
    pitch = np.degrees(np.arctan2(-pencil_dir[1], np.sqrt(pencil_dir[0]**2 + pencil_dir[2]**2)))
    
    # Compute palm normal via cross product from wrist, index_mcp, and pinky_mcp.
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    palm_normal = normalize(np.cross(v1, v2))
    
    # Project the palm normal onto the plane perpendicular to pencil_dir.
    proj_palm = palm_normal - np.dot(palm_normal, pencil_dir) * pencil_dir
    proj_palm = normalize(proj_palm)
    
    # Define global up (in camera coords, assume upward is negative y).
    global_up = np.array([0, -1, 0])
    # Project global up onto the plane perpendicular to pencil_dir.
    proj_up = global_up - np.dot(global_up, pencil_dir) * pencil_dir
    proj_up = normalize(proj_up)
    
    # Compute roll as the signed angle between proj_up and proj_palm.
    # The angle between two vectors:
    dot_val = np.clip(np.dot(proj_palm, proj_up), -1.0, 1.0)
    roll = np.degrees(np.arccos(dot_val))
    # Determine sign: use cross product and check direction relative to pencil_dir.
    cross_val = np.cross(proj_up, proj_palm)
    if np.dot(cross_val, pencil_dir) < 0:
        roll = -roll

    return pitch, yaw, roll, pencil_dir

# ---------- Main Application ----------

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    time.sleep(2)
    print("Webcam stream opened successfully.")
    
    # Grab one frame to get dimensions.
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from camera.")
        return
    h, w, _ = frame.shape
    # For drawing, compute frame center in pixels.
    center_px = (w // 2, h // 2)
    
    # Variables for exponential smoothing.
    smoothed = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    alpha = 0.3  # smoothing factor

    # Calibration: when the pencil is held in the ideal (upright) orientation,
    # press 'c' to capture neutral values.
    ref_angles = None  # (ref_pitch, ref_yaw, ref_roll)
    
    # Simulated servo angles.
    current_pitch = 90
    current_yaw   = 90
    current_roll  = 90
    
    # Gains for servo updates.
    pitch_gain = 0.05
    yaw_gain   = 0.05
    roll_gain  = 0.1
    
    print("Press 'c' to calibrate (set current orientation as neutral).")
    
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
                
                # For visualization, draw all landmarks.
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Compute Euler angles from our 3D model.
                pitch, yaw, roll, pencil_dir = compute_euler_angles(hand_landmarks, w, h)
                
                # Apply exponential smoothing.
                smoothed["pitch"] = exponential_smoothing(pitch, smoothed["pitch"], alpha)
                smoothed["yaw"]   = exponential_smoothing(yaw,   smoothed["yaw"],   alpha)
                smoothed["roll"]  = exponential_smoothing(roll,  smoothed["roll"],  alpha)
                
                # If calibration is done, compute errors relative to reference.
                if ref_angles is not None:
                    error_pitch = smoothed["pitch"] - ref_angles[0]
                    error_yaw   = smoothed["yaw"]   - ref_angles[1]
                    error_roll  = smoothed["roll"]  - ref_angles[2]
                else:
                    error_pitch = smoothed["pitch"]
                    error_yaw   = smoothed["yaw"]
                    error_roll  = smoothed["roll"]
                
                # Update servo angles based on errors.
                current_pitch = set_virtual_servo_angle(current_pitch, error_pitch, pitch_gain)
                current_yaw   = set_virtual_servo_angle(current_yaw,   error_yaw,   yaw_gain)
                current_roll  = set_virtual_servo_angle(current_roll,  error_roll,  roll_gain)
                
                # Display the raw and smoothed angles (for debugging).
                debug_text = f"Raw: P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f} | Smth: P:{smoothed['pitch']:.1f} Y:{smoothed['yaw']:.1f} R:{smoothed['roll']:.1f}"
                cv2.putText(frame, debug_text, (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                # No hand detected; optionally, you might want to hold or reset servo angles.
                error_pitch = error_yaw = error_roll = 0

            # Display servo angles on the top left (stacked).
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
                cv2.putText(frame, f"(Calibrated)", (10, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            cv2.imshow("Advanced 3D Hand Orientation", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Calibrate: store current smoothed values as neutral.
                ref_angles = (smoothed["pitch"], smoothed["yaw"], smoothed["roll"])
                print(f"Calibrated: Pitch {ref_angles[0]:.2f}, Yaw {ref_angles[1]:.2f}, Roll {ref_angles[2]:.2f}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
