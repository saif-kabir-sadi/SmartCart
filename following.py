import cv2
import numpy as np
import time
import serial
from picamera2 import Picamera2


# ==================================================
# ESP32 SERIAL SETUP
# ==================================================

ESP32_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

try:
    esp32 = serial.Serial(
        ESP32_PORT,
        BAUD_RATE,
        timeout=1
    )
    time.sleep(2)
    print("ESP32 connected:", ESP32_PORT)
except Exception as e:
    print("Failed to connect to ESP32:", e)
    exit(1)


# ==================================================
# SETTINGS & RESOLUTION
# ==================================================

MARKER_ID = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Offset in pixels from screen center for left/right limits
LIMIT_OFFSET = 90


# ==================================================
# DISTANCE SETTINGS
# ==================================================

STOP_DISTANCE = 0.50
FOLLOW_DISTANCE = 1.20


# ==================================================
# CAMERA CALIBRATION
# ==================================================

data = np.load("camera_calibration.npz")

camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]


# ==================================================
# ARUCO DETECTOR
# ==================================================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)


# ==================================================
# PI CAMERA SETUP (Raspberry Pi 5 / PiSP Safe Setup)
# ==================================================

try:
    picam2 = Picamera2(camera_num=1)
    
    # Disabling raw stream prevents /dev/video4-6 buffer locks
    config = picam2.create_preview_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
        raw=None
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("Pi Camera initialized successfully.")
except Exception as e:
    print("Failed to initialize Pi Camera:", e)
    exit(1)


# ==================================================
# MARKER 3D POINTS
# ==================================================

MARKER_SIZE = 0.06  # Marker side length in meters

half = MARKER_SIZE / 2.0

object_points = np.array(
    [
        [-half, half, 0],
        [half, half, 0],
        [half, -half, 0],
        [-half, -half, 0]
    ],
    dtype=np.float32
)


# ==================================================
# COMMAND SETTINGS
# ==================================================

COMMAND_INTERVAL = 0.05  # Send every 50 ms

last_command = ""
last_command_time = 0


# ==================================================
# SERIAL COMMAND FUNCTIONS
# ==================================================

def send_command(command):
    global last_command

    command = command.strip().upper()

    try:
        esp32.write((command + "\n").encode())
        esp32.flush()

        if command != last_command:
            print(">>> ESP32 COMMAND CHANGED:", command)
            last_command = command

    except serial.SerialException as e:
        print("ESP32 SERIAL ERROR:", e)


def update_motor_command(action):
    global last_command_time

    current_time = time.time()

    if action != last_command:
        send_command(action)
        last_command_time = current_time

    elif current_time - last_command_time >= COMMAND_INTERVAL:
        send_command(action)
        last_command_time = current_time


# ==================================================
# STARTUP MESSAGE
# ==================================================

print()
print("==========================================")
print("MY TROLLEY - ARUCO FOLLOW SYSTEM (PI CAMERA)")
print("==========================================")
print()

print("Marker ID        :", MARKER_ID)
print("STOP distance    :", STOP_DISTANCE, "m")
print("Follow distance :", FOLLOW_DISTANCE, "m")
print("Command interval:", COMMAND_INTERVAL, "seconds")
print("Command rate     :", int(1 / COMMAND_INTERVAL), "commands/sec")

print()
print("Press Q to quit or Ctrl+C to stop.")
print()


# ==================================================
# MAIN LOOP
# ==================================================

try:
    while True:

        # CAPTURE FRAME FROM PI CAMERA
        raw_frame = picam2.capture_array()

        if raw_frame is None:
            continue

        # CONVERT RGB TO BGR FOR OPENCV
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

        # DYNAMIC FRAME SIZE & CENTER CALCULATION
        actual_height, actual_width = frame.shape[:2]
        center_screen_x = actual_width // 2

        # DYNAMIC LEFT / RIGHT LIMITS
        LEFT_LIMIT = center_screen_x - LIMIT_OFFSET
        RIGHT_LIMIT = center_screen_x + LIMIT_OFFSET

        # CONVERT TO GRAYSCALE
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ARUCO DETECTION
        corners, ids, rejected = detector.detectMarkers(gray)

        # COPY FRAME FOR ANNOTATED DISPLAY
        display = frame.copy()

        # DRAW DYNAMIC CENTER (CYAN) & BOUNDARY LINES (YELLOW)
        cv2.line(display, (center_screen_x, 0), (center_screen_x, actual_height), (255, 255, 0), 2)
        cv2.line(display, (LEFT_LIMIT, 0), (LEFT_LIMIT, actual_height), (0, 255, 255), 2)
        cv2.line(display, (RIGHT_LIMIT, 0), (RIGHT_LIMIT, actual_height), (0, 255, 255), 2)

        marker_found = False
        action = "STOP"
        direction = "NO MARKER"

        # CHECK DETECTED MARKERS
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()

            for i, current_id in enumerate(ids_flat):

                if int(current_id) != MARKER_ID:
                    continue

                marker_found = True

                # GET CORNERS
                image_points = corners[i][0].astype(np.float32)

                # MARKER CENTER
                center_x = int(np.mean(image_points[:, 0]))
                center_y = int(np.mean(image_points[:, 1]))

                # DRAW MARKER OUTLINE & CENTER POINT
                cv2.polylines(
                    display,
                    [image_points.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    2
                )
                cv2.circle(display, (center_x, center_y), 8, (0, 0, 255), -1)

                # POSE ESTIMATION & DISTANCE CALCULATION
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    image_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if not success:
                    print("solvePnP failed")
                    action = "STOP"
                    update_motor_command(action)
                    break

                x = float(tvec[0][0])
                y = float(tvec[1][0])
                z = float(tvec[2][0])

                distance = float(np.sqrt(x * x + y * y + z * z))

                # DIRECTION DETERMINATION
                if center_x < LEFT_LIMIT:
                    direction = "LEFT"
                elif center_x > RIGHT_LIMIT:
                    direction = "RIGHT"
                else:
                    direction = "CENTER"

                # MOVEMENT DECISION
                if distance <= STOP_DISTANCE:
                    action = "STOP"
                elif center_x < LEFT_LIMIT:
                    action = "LEFT"
                elif center_x > RIGHT_LIMIT:
                    action = "RIGHT"
                else:
                    action = "FORWARD"

                # TRANSMIT COMMAND TO ESP32
                update_motor_command(action)

                # OVERLAY DATA ON SCREEN
                cv2.putText(
                    display,
                    f"Distance: {distance:.2f} m",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    display,
                    f"X: {center_x}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    display,
                    f"Direction: {direction}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    display,
                    f"ACTION: {action}",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2
                )

                print(
                    f"X={center_x} | "
                    f"Distance={distance:.2f}m | "
                    f"Direction={direction} | "
                    f"Action={action}"
                )

                break

        # NO MARKER FOUND IN CURRENT FRAME
        if not marker_found:
            action = "STOP"
            update_motor_command(action)

            cv2.putText(
                display,
                "NO MARKER -> STOP",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        # SHOW DISPLAY WINDOW
        cv2.imshow("MyTrolley Follow Decision", display)

        # EXIT CONDITION
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

except KeyboardInterrupt:
    print("\nProgram interrupted by user.")

finally:
    print("\nCleaning up resources...")

    # 1. RELEASE PI CAMERA FIRST (Prevents V4L2 lock even on multiple Ctrl+C)
    if 'picam2' in locals():
        try:
            picam2.stop()
            picam2.close()
            print("Pi Camera stopped cleanly.")
        except Exception as e:
            print("Camera release warning:", e)

    # 2. FORCE STOP COMMAND TO ESP32 AND CLOSE SERIAL
    if 'esp32' in locals():
        try:
            esp32.write(b"STOP\n")
            esp32.flush()
            try:
                time.sleep(0.2)
            except BaseException:
                pass
            esp32.close()
            print("ESP32 serial closed.")
        except Exception as e:
            print("ESP32 serial warning:", e)

    cv2.destroyAllWindows()
    print("Program stopped safely.")
