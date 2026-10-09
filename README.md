# MyTrolley 🛒

A smart, self-following shopping trolley built on a **Raspberry Pi 5** and an **ESP32**. The trolley follows the customer using an ArUco marker, scans product barcodes, answers questions by voice, shows a touch-screen UI with checkout and QR payment, and opens a gate when payment is complete.

---

## Features

| Feature | How it works |
|---|---|
| **Follow-me navigation** | Pi Camera detects an ArUco marker (ID 0), estimates distance with `solvePnP`, and sends `FORWARD` / `LEFT` / `RIGHT` / `STOP` to the ESP32 over serial. |
| **Barcode scanning** | Camera 0 reads barcodes with `pyzbar`; known barcodes are added to the cart and unknown ones are ignored. |
| **Product counting + belt control** | A USB camera runs YOLO-World (`yolov8s-world.pt`); when exactly one item is detected the conveyor belt (relay + servo) starts, otherwise it stops (debounced over 3 frames). |
| **Voice assistant** | Push-to-talk (touch button or GPIO button) → `faster-whisper` (`tiny.en`) speech-to-text → Ollama (`gemma3:1b`) answers using the store inventory only → `pocket-tts` speaks the reply. |
| **Touch-screen UI** | ILI9341 240×320 SPI display with XPT2046 touch: Main, Speak, Checkout, Details, Payment, Thank You screens. |
| **Checkout & payment** | Generates a payment QR code, reduces inventory on confirmation, clears the cart, and opens/closes the gate servos. |
| **Safe shutdown** | Ctrl+C / SIGTERM stops the relay, frees GPIO/SPI, and sends `STOP` to the ESP32. |

---

## Project Structure

```
MyTrolley/
├── main.py                  # Main system: UI, barcode, voice assistant, belt, gate, payment
├── following.py             # Standalone ArUco follow-me script (Pi Camera + ESP32)
├── start.sh                 # Launches main.py and following.py together
├── requirements.txt         # Python packages installed on the Pi
├── camera_calibration.npz   # Camera calibration data (required, you must generate this)
├── apple_logo.jpg           # Logo used on the display
└── yolov8s-world.pt         # YOLO-World weights
```

> `camera_calibration.npz`, `apple_logo.jpg`, and `yolov8s-world.pt` are loaded at runtime and must be present in the project folder.

---

## Hardware

- Raspberry Pi 5
- Pi Camera(s) (via `picamera2`) + 1 USB camera (for item counting)
- ESP32 (motor controller, connected over USB serial)
- ILI9341 TFT display (240×320, SPI) with XPT2046 touch controller
- Relay module + gear motor (conveyor belt)
- Servo motors: 1 for the belt, 2 for the gate doors
- Push button, USB microphone, speaker

### Pin Map (GPIO numbering)

| Function | Pin |
|---|---|
| Gate servo 1 | GPIO 2 |
| Gate servo 2 | GPIO 3 |
| Belt relay | GPIO 5 (active LOW) |
| Belt servo | GPIO 6 |
| Voice push button | GPIO 17 |
| Display DC | GPIO 25 |
| Display RESET | GPIO 24 |
| Display CS / Touch CS | SPI0 CE0 / CE1 |

### Serial Ports

| Script | ESP32 port |
|---|---|
| `main.py` | `/dev/ttyUSB0` |
| `following.py` | `/dev/ttyACM0` |

Baud rate is `115200`. The ESP32 firmware is **not included** in this repository. It must accept newline-terminated text commands: `FORWARD`, `LEFT`, `RIGHT`, `STOP`.

---

## Installation

```bash
cd ~/MyTrolley
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: `opencv-python`, `numpy`, `picamera2`, `pyserial`, `ultralytics`, `faster-whisper`, `pocket-tts`, `ollama`, `sounddevice`, `pyzbar`, `qrcode`, `spidev`, `lgpio`, `gpiozero`, `Pillow`.

> `requirements.txt` is a full `pip list` dump from the Pi, so it contains many system packages that are not needed by this project. Trim it if you want a lighter install. `picamera2`, `lgpio`, and `spidev` are best installed through `apt` / the system Python.

### Ollama model

Install [Ollama](https://ollama.com) and pull the model used by the assistant:

```bash
ollama pull gemma3:1b
```

### Camera calibration

`following.py` and `main.py` need `camera_calibration.npz` containing `camera_matrix` and `dist_coeffs`. Generate it with a standard OpenCV chessboard calibration for your camera at 1280×720.

---

## Running

```bash
./start.sh
```

`start.sh` activates the virtual environment, kills any old Python processes, then starts both scripts in the background:

```bash
python main.py &
python following.py &
```

You can also run them individually:

```bash
python main.py        # full system
python following.py   # follow-me only (press Q in the preview window to quit)
```

> ⚠️ `start.sh` runs `sudo pkill -9 -f python`, which kills **every** Python process on the Pi. Use with care.

---

## Follow-Me Settings

Edit these constants in `following.py`:

| Setting | Default | Meaning |
|---|---|---|
| `MARKER_ID` | `0` | ArUco ID to follow (`DICT_4X4_50`) |
| `MARKER_SIZE` | `0.06` m | Printed marker side length |
| `STOP_DISTANCE` | `0.50` m | Trolley stops at or below this distance |
| `LIMIT_OFFSET` | `90` px | Dead zone around frame center before turning |
| `COMMAND_INTERVAL` | `0.05` s | How often commands are re-sent to the ESP32 |

**Decision logic:** marker lost → `STOP`; distance ≤ stop distance → `STOP`; marker left of the dead zone → `LEFT`; right of it → `RIGHT`; otherwise → `FORWARD`.

---

## Store Inventory

The inventory lives in the `SHOP_DATA` dictionary in `main.py`. Each item has a product name, barcode, available quantity, unit price, and shelf location (rack / row / shelf):

```python
{
    "product": "Rice",
    "barcode": "1111111111",
    "available_quantity": "10 kg",
    "unit_price": "$1.5 per kg",
    "exact_location": {"rack": 2, "row": 1, "shelf": 2}
}
```

The voice assistant answers **only** from this table. Stock is reduced automatically after a payment is confirmed.

---

## UI Flow

```
MAIN ──► SPEAK (talk to assistant)
  │
  └───► CHECKOUT ──► DETAILS (cart items)
            │
            └──► PAYMENT (QR) ──► THANK YOU ──► MAIN
                    (gate opens for 5 s, then closes)
```

---

## Known Notes

- `main.py` and `following.py` both drive the ESP32 and cameras. `main.py` already contains its own ArUco follow logic in `run_camera_0`, so running both at the same time may conflict over the camera and motor commands. Run only the one you need unless the cameras and serial ports are separate.
- The payment flow generates a QR code with a random `MYTROLLEY-PAY-XXXX` reference. The **PAID** button confirms payment manually; there is no real payment gateway integration.
- Prices and the store name are demo data.

---

## License

Add your license here.
