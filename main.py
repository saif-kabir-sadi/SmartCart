import os
import sys
import time
import signal
import threading
import serial
import cv2
import numpy as np
import spidev
import lgpio
import uuid
import qrcode
from PIL import Image, ImageDraw, ImageFont, ImageOps
from picamera2 import Picamera2
from pyzbar.pyzbar import decode
from ultralytics import YOLOWorld
import sounddevice as sd
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel
from pocket_tts import TTSModel
from ollama import chat
from gpiozero import Button


# =========================================================
# GATE OPEN AND CLOSING
# =========================================================
from gpiozero import Servo
from time import sleep

SERVO_1_PIN = 2  # Physical Pin 3
SERVO_2_PIN = 3  # Physical Pin 5

servo1 = Servo(SERVO_1_PIN, initial_value=None, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)
servo2 = Servo(SERVO_2_PIN, initial_value=None, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

def open_doors():
    # Full 90 degree position-e khulbe
    servo1.value = -1.0  # 0 degree
    servo2.value = 0.0   # 90 degree

def close_doors():
    # Full 90 degree position-e bondho hobe (0.0 ebong -1.0)
    servo1.value = 0.0   # 90 degree
    servo2.value = -1.0  # 0 degree
    sleep(0.8)
    
    # Off hone por signal detach kora jate motor garam na hoy
    servo1.detach()
    servo2.detach()


# =========================================================
# BELT RUNNING CODE (FIXED)
# =========================================================
from gpiozero import OutputDevice, Servo
from time import sleep

# Pin Definitions (GPIO numbering)
RELAY_PIN = 5   # GPIO 5
SERVO_PIN = 6   # GPIO 6

# রিলে মডিউল সেটআপ (Active LOW হলে active_high=False রাখুন)
# আপনার রিলে Active HIGH হলে active_high=True করে দিন।
relay = OutputDevice(RELAY_PIN, active_high=False, initial_value=False)

# সারভো মোটর সেটআপ (SG90/MG996R এর জন্য pulse width ফিক্স করা)
servo = Servo(SERVO_PIN, initial_value=None, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

def set_servo_angle(angle):
    """
    অ্যাঙ্গেল (০ থেকে ১৮০ ডিগ্রী) সেট করার ফাংশন।
    gpiozero-তে সারভোর ভ্যালু -১ (০°) থেকে +১ (১৮০°) পর্যন্ত হয়।
    """
    value = (angle / 90.0) - 1.0  # angle কে -1.0 থেকে +1.0 এ রূপান্তর
    servo.value = value
    sleep(0.4)
    servo.detach()  # সারভোর কাঁপুনি (jitter) এবং হিট হওয়া বন্ধ করার জন্য সিগন্যাল অফ

def start_belt():
    print("Belt level: STARTING (Gear motor ON & Servo moving to 90°)...")
    relay.on()            # গিয়ার মোটর চালু
    set_servo_angle(90)   # সারভো ৯০ ডিগ্রী-তে নেওয়া

def stop_belt():
    print("Belt level: STOPPING (Gear motor OFF & Servo reset to 0°)...")
    relay.off()           # গিয়ার মোটর বন্ধ
    set_servo_angle(0)    # সারভো ০ ডিগ্রী-তে ফেরত আনা


# Belt er current state + debounce counter
belt_running = False
belt_on_streak = 0
belt_off_streak = 0
BELT_DEBOUNCE_FRAMES = 3   # ek-i result koto frame porpor holey state change hobe


# =========================================================
# GLOBAL CONSTANTS & SHARED DATA
# =========================================================
PROGRAM_START = time.time()

# Shared Camera 1 State (YOLO Segmentation)
total_products_count = 0
counted_ids_set = set()

# Shared Camera 0 State (Barcode & ArUco Follow System)
last_detected_barcode = ""
last_barcode_time = 0
current_aruco_action = "STOP"

# =========================================================
# SCANNED PRODUCTS
# =========================================================
# Stores only the products whose barcodes were scanned.
#
# Example:
# {
#     "1111111111": {
#         "product": "Rice",
#         "barcode": "1111111111",
#         "unit_price": "$1.5 per kg",
#         "quantity": 2
#     }
# }
scanned_products = {}

# UI & Voice State Management
current_ui_state = "MAIN"
# "MAIN", "SPEAK", "CHECKOUT", "DETAILS", "PAYMENT", "THANK_YOU"

voice_assistant_status = "READY"
# "READY", "LISTENING", "THINKING", "SPEAKING"

# Payment QR
payment_qr_image = None
payment_qr_data = ""

# Thank You Screen
thank_you_until = 0

# Lock for Thread-Safe Global State Updates
state_lock = threading.Lock()

# =========================================================
# LOGGING HELPER
# =========================================================
def log(msg):
    elapsed = int(time.time() - PROGRAM_START)
    print(f"[{elapsed}s] {msg}", flush=True)

# =========================================================
# GPIO & SPI INITIALIZATION (ILI9341 & XPT2046)
# =========================================================
WIDTH = 240
HEIGHT = 320

LCD_CS = 0
TOUCH_CS = 1

DC_PIN = 25
RESET_PIN = 24
BUTTON_PIN = 17

LCD_SPEED = 16_000_000
TOUCH_SPEED = 1_000_000

# Color Palette
PINK_BG = (255, 210, 225)
DARK_PINK = (210, 50, 110)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APPLE_LOGO = os.path.join(BASE_DIR, "apple_logo.jpg")

gpio_chip = None
for c_num in [0, 4]:
    try:
        gpio_chip = lgpio.gpiochip_open(c_num)
        log(f"Successfully opened GPIO chip {c_num}")
        break
    except Exception:
        pass

if gpio_chip is None:
    log("ERROR: Could not open any GPIO chip.")
    sys.exit(1)

for pin in [DC_PIN, RESET_PIN]:
    try:
        lgpio.gpio_free(gpio_chip, pin)
    except Exception:
        pass

lgpio.gpio_claim_output(gpio_chip, DC_PIN, 0)
lgpio.gpio_claim_output(gpio_chip, RESET_PIN, 1)

# SPI Setup
lcd_spi = spidev.SpiDev()
lcd_spi.open(0, LCD_CS)
lcd_spi.max_speed_hz = LCD_SPEED
lcd_spi.mode = 0

touch_spi = spidev.SpiDev()
touch_spi.open(0, TOUCH_CS)
touch_spi.max_speed_hz = TOUCH_SPEED
touch_spi.mode = 0


def lcd_cmd(value):
    lgpio.gpio_write(gpio_chip, DC_PIN, 0)
    lcd_spi.writebytes([value])


def lcd_data(values):
    lgpio.gpio_write(gpio_chip, DC_PIN, 1)

    if isinstance(values, int):
        values = [values]

    for i in range(0, len(values), 4096):
        lcd_spi.writebytes(values[i:i + 4096])


def reset_display():
    lgpio.gpio_write(gpio_chip, RESET_PIN, 1)
    time.sleep(0.05)

    lgpio.gpio_write(gpio_chip, RESET_PIN, 0)
    time.sleep(0.1)

    lgpio.gpio_write(gpio_chip, RESET_PIN, 1)
    time.sleep(0.15)


def init_display():
    log("Initializing Display...")

    reset_display()

    lcd_cmd(0x01)
    time.sleep(0.15)

    lcd_cmd(0x11)
    time.sleep(0.12)

    lcd_cmd(0x3A)
    lcd_data(0x55)

    lcd_cmd(0x36)
    lcd_data(0x48)

    lcd_cmd(0xB1)
    lcd_data([0x00, 0x18])

    lcd_cmd(0xB6)
    lcd_data([0x08, 0x82, 0x27])

    lcd_cmd(0xC0)
    lcd_data(0x23)

    lcd_cmd(0xC1)
    lcd_data(0x10)

    lcd_cmd(0xC5)
    lcd_data([0x3E, 0x28])

    lcd_cmd(0xE0)
    lcd_data([
        0x0F, 0x31, 0x2B, 0x0C, 0x0E,
        0x08, 0x4E, 0xF1, 0x37, 0x07,
        0x10, 0x03, 0x0E, 0x09, 0x00
    ])

    lcd_cmd(0xE1)
    lcd_data([
        0x00, 0x0E, 0x14, 0x03, 0x11,
        0x07, 0x31, 0xC1, 0x48, 0x08,
        0x0F, 0x0C, 0x31, 0x36, 0x0F
    ])

    lcd_cmd(0x29)
    time.sleep(0.1)

    log("Display Initialized.")


def set_window(x0, y0, x1, y1):
    lcd_cmd(0x2A)

    lcd_data([
        (x0 >> 8) & 0xFF,
        x0 & 0xFF,
        (x1 >> 8) & 0xFF,
        x1 & 0xFF
    ])

    lcd_cmd(0x2B)

    lcd_data([
        (y0 >> 8) & 0xFF,
        y0 & 0xFF,
        (y1 >> 8) & 0xFF,
        y1 & 0xFF
    ])

    lcd_cmd(0x2C)


def show_image(image):
    image = image.convert("RGB")

    if image.size != (WIDTH, HEIGHT):
        image = image.resize(
            (WIDTH, HEIGHT),
            Image.Resampling.LANCZOS
        )

    set_window(
        0,
        0,
        WIDTH - 1,
        HEIGHT - 1
    )

    pixels = np.asarray(image)

    for y in range(HEIGHT):
        row = pixels[y]

        r = row[:, 0].astype(np.uint16)
        g = row[:, 1].astype(np.uint16)
        b = row[:, 2].astype(np.uint16)

        rgb565 = (
            ((r & 0xF8) << 8)
            | ((g & 0xFC) << 3)
            | (b >> 3)
        )

        row_bytes = rgb565.astype(">u2").tobytes()

        lgpio.gpio_write(
            gpio_chip,
            DC_PIN,
            1
        )

        lcd_spi.writebytes(
            list(row_bytes)
        )

def show_barcode_modal(barcode, product_name="Unknown Product"):
    """
    Scanned Barcode and Product Name Modal Display.
    Fully self-contained: Temporarily overrides show_image so main loop renders 
    the modal for 2 seconds without any SPI corruption or needing to edit main().
    """
    global show_image  # Global show_image function reference

    # 1. Base Image background
    img = Image.new("RGB", (WIDTH, HEIGHT), PINK_BG)
    draw = ImageDraw.Draw(img)

    # 2. Top Header Bar
    draw.rectangle([0, 0, WIDTH, 45], fill=DARK_PINK)
    draw_center_text(draw, "MyTrolley", 12, FONT_MEDIUM, WHITE)

    # 3. Main Modal Card (Center Box)
    draw.rounded_rectangle([15, 75, 225, 255], radius=15, fill=WHITE, outline=DARK_PINK, width=3)

    # 4. Badge Title Box
    draw.rounded_rectangle([30, 60, 210, 92], radius=10, fill=DARK_PINK)
    draw_button_text(draw, "BARCODE SCANNED", [30, 60, 210, 92], FONT_SMALL, WHITE)

    # 5. Product Name Formatting
    if len(product_name) > 18:
        product_name = product_name[:16] + ".."

    # 6. Text Content Inside Box
    draw_center_text(draw, f"Item: {product_name}", 125, FONT_MEDIUM, BLACK)
    draw_center_text(draw, f"Code: {barcode}", 165, FONT_SMALL, DARK_PINK)
    draw_center_text(draw, "✓ Added to Cart", 205, FONT_SMALL, BLACK)

    # 7. TEMPORARILY OVERRIDE SHOW_IMAGE (Prevents SPI Collision)
    original_show_image = show_image

    def modal_override_show_image(ignored_img):
        # Force main loop to draw this modal image instead of its own screens
        original_show_image(img)

    try:
        # Patch show_image for main thread
        show_image = modal_override_show_image
        
        # Hold modal for exactly 2 seconds
        time.sleep(2.0)

    finally:
        # Restore original show_image function safely
        show_image = original_show_image

# =========================================================
# TOUCH READER (XPT2046)
# =========================================================
def read_touch_raw():
    rx = touch_spi.xfer2([
        0xD0,
        0x00,
        0x00
    ])

    ry = touch_spi.xfer2([
        0x90,
        0x00,
        0x00
    ])

    x = ((rx[1] << 8) | rx[2]) >> 3
    y = ((ry[1] << 8) | ry[2]) >> 3

    return x, y


def read_touch():
    xs = []
    ys = []

    for _ in range(3):
        x, y = read_touch_raw()

        if (
            100 < x < 4000
            and
            100 < y < 4000
        ):
            xs.append(x)
            ys.append(y)

    if len(xs) < 2:
        return None

    x_avg = sum(xs) // len(xs)
    y_avg = sum(ys) // len(ys)

    screen_x = int(
        (x_avg - 200)
        * WIDTH
        /
        (3900 - 200)
    )

    # Correct touchscreen Y-axis
    screen_y = HEIGHT - 1 - int(
        (y_avg - 200)
        * HEIGHT
        /
        (3900 - 200)
    )

    screen_x = max(
        0,
        min(WIDTH - 1, screen_x)
    )

    screen_y = max(
        0,
        min(HEIGHT - 1, screen_y)
    )

    return screen_x, screen_y


# =========================================================
# FONT DEFINITIONS
# =========================================================
try:
    FONT_LARGE = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        22
    )

    FONT_MEDIUM = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        16
    )

    FONT_SMALL = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        12
    )

except Exception:
    FONT_LARGE = FONT_MEDIUM = FONT_SMALL = ImageFont.load_default()


# =========================================================
# BOOT SCREEN LOGIC
# =========================================================
def boot_screen():
    log("Running Apple Boot Screen...")

    if os.path.exists(APPLE_LOGO):
        try:
            logo = Image.open(
                APPLE_LOGO
            ).convert("RGB")

            gray = ImageOps.grayscale(logo)

            logo = gray.point(
                lambda p: 255 if p < 150 else 0
            ).convert("RGB")

            logo.thumbnail(
                (150, 150),
                Image.Resampling.LANCZOS
            )

            image = Image.new(
                "RGB",
                (WIDTH, HEIGHT),
                BLACK
            )

            x = (WIDTH - logo.width) // 2
            y = (HEIGHT - logo.height) // 2

            image.paste(
                logo,
                (x, y)
            )

            show_image(image)

            time.sleep(0.5)

            return

        except Exception as e:
            log(f"Boot Screen Error: {e}")

    image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        BLACK
    )

    show_image(image)

    time.sleep(2)


# =========================================================
# UI DRAWING FUNCTIONS
# =========================================================
def draw_center_text(
    draw,
    text,
    y,
    font,
    fill
):
    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    tw = bbox[2] - bbox[0]

    draw.text(
        (
            (WIDTH - tw) // 2,
            y
        ),
        text,
        font=font,
        fill=fill
    )


# =========================================================
# NEW: DRAW TEXT CENTERED INSIDE A BUTTON
# =========================================================
def draw_button_text(
    draw,
    text,
    box,
    font,
    fill
):
    x1, y1, x2, y2 = box

    bbox = draw.textbbox(
        (0, 0),
        text,
        font=font
    )

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    draw.text(
        (
            cx - tw // 2 - bbox[0],
            cy - th // 2 - bbox[1]
        ),
        text,
        font=font,
        fill=fill
    )


def render_main_menu():
    img = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        PINK_BG
    )

    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [0, 0, WIDTH, 50],
        fill=DARK_PINK
    )

    draw_center_text(
        draw,
        "MyTrolley",
        14,
        FONT_LARGE,
        WHITE
    )

    draw.rounded_rectangle(
        [30, 90, 210, 160],
        radius=15,
        fill=DARK_PINK
    )

    draw_center_text(
        draw,
        "1. SPEAK",
        112,
        FONT_LARGE,
        WHITE
    )

    draw.rounded_rectangle(
        [30, 190, 210, 260],
        radius=15,
        fill=DARK_PINK
    )

    draw_center_text(
        draw,
        "2. CHECKOUT",
        212,
        FONT_LARGE,
        WHITE
    )

    return img


def render_speak_screen(
    status="READY",
    frame_num=0
):
    img = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        PINK_BG
    )

    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [0, 0, WIDTH, 45],
        fill=DARK_PINK
    )

    draw_center_text(
        draw,
        "Voice Assistant",
        12,
        FONT_MEDIUM,
        WHITE
    )

    cx = WIDTH // 2
    cy = 115

    if status == "READY":

        draw.ellipse(
            [
                cx - 35,
                cy - 35,
                cx + 35,
                cy + 35
            ],
            fill=DARK_PINK
        )

        draw_center_text(
            draw,
            "READY",
            165,
            FONT_MEDIUM,
            DARK_PINK
        )

        draw_center_text(
            draw,
            "Tap 'TALK' or Button",
            190,
            FONT_SMALL,
            DARK_PINK
        )

    elif status == "LISTENING":

        draw_center_text(
            draw,
            "LISTENING...",
            165,
            FONT_MEDIUM,
            DARK_PINK
        )

        bar_w = 10
        gap = 6

        start_x = cx - (
            (5 * bar_w + 4 * gap) // 2
        )

        for i in range(5):

            h = max(
                10,
                25 + int(
                    20
                    *
                    np.sin(
                        frame_num * 0.5
                        +
                        i * 0.8
                    )
                )
            )

            bx = start_x + i * (
                bar_w + gap
            )

            draw.rounded_rectangle(
                [
                    bx,
                    cy - h // 2,
                    bx + bar_w,
                    cy + h // 2
                ],
                radius=4,
                fill=DARK_PINK
            )

    elif status == "THINKING":

        draw_center_text(
            draw,
            "THINKING...",
            165,
            FONT_MEDIUM,
            DARK_PINK
        )

        for i in range(3):

            offset = int(
                12
                *
                abs(
                    np.sin(
                        frame_num * 0.4
                        +
                        i * 1.2
                    )
                )
            )

            bx = cx - 35 + i * 35

            draw.ellipse(
                [
                    bx - 8,
                    cy - 8 - offset,
                    bx + 8,
                    cy + 8 - offset
                ],
                fill=DARK_PINK
            )

    elif status == "SPEAKING":

        draw_center_text(
            draw,
            "SPEAKING...",
            165,
            FONT_MEDIUM,
            DARK_PINK
        )

        draw.rectangle(
            [
                cx - 15,
                cy - 15,
                cx + 5,
                cy + 15
            ],
            fill=DARK_PINK
        )

        draw.polygon(
            [
                (cx + 5, cy - 15),
                (cx + 20, cy - 25),
                (cx + 20, cy + 25),
                (cx + 5, cy + 15)
            ],
            fill=DARK_PINK
        )

        wave_offset = (
            frame_num * 3
        ) % 30

        for i in range(2):

            rad = (
                25
                +
                i * 18
                +
                wave_offset
            )

            if rad < 60:

                draw.arc(
                    [
                        cx - rad,
                        cy - rad,
                        cx + rad,
                        cy + rad
                    ],
                    -50,
                    50,
                    fill=DARK_PINK,
                    width=3
                )

    # =====================================================
    # TALK BUTTON
    # =====================================================
    talk_box = [20, 215, 115, 260]

    draw.rounded_rectangle(
        talk_box,
        radius=10,
        fill=DARK_PINK
    )

    draw_button_text(
        draw,
        "TALK",
        talk_box,
        FONT_MEDIUM,
        WHITE
    )

    # =====================================================
    # BACK BUTTON
    # =====================================================
    back_box = [125, 215, 220, 260]

    draw.rounded_rectangle(
        back_box,
        radius=10,
        fill=BLACK
    )

    draw_button_text(
        draw,
        "BACK",
        back_box,
        FONT_MEDIUM,
        WHITE
    )

    return img


# =========================================================
# CHECKOUT SCREEN
# =========================================================
def render_checkout_screen():

    img = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        PINK_BG
    )

    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [0, 0, WIDTH, 50],
        fill=DARK_PINK
    )

    draw_center_text(
        draw,
        "Checkout Summary",
        14,
        FONT_LARGE,
        WHITE
    )

    draw_center_text(
        draw,
        "SUMMARY",
        75,
        FONT_MEDIUM,
        DARK_PINK
    )

    draw.rectangle(
        [20, 105, 220, 205],
        fill=WHITE,
        outline=DARK_PINK,
        width=2
    )

    draw.text(
        (30, 120),
        f"Items Count: {total_products_count}",
        font=FONT_MEDIUM,
        fill=BLACK
    )

    b_code = (
        last_detected_barcode
        if last_detected_barcode
        else "None"
    )

    if len(b_code) > 14:
        b_code = b_code[:12] + ".."

    draw.text(
        (30, 150),
        f"Barcode: {b_code}",
        font=FONT_SMALL,
        fill=BLACK
    )

    draw.text(
        (30, 175),
        "Status: Active",
        font=FONT_MEDIUM,
        fill=DARK_PINK
    )

    # =====================================================
    # PAY BUTTON
    # =====================================================
    pay_box = [10, 220, 75, 270]

    draw.rounded_rectangle(
        pay_box,
        radius=10,
        fill=DARK_PINK
    )

    draw_button_text(
        draw,
        "PAY",
        pay_box,
        FONT_SMALL,
        WHITE
    )

    # =====================================================
    # DETAILS BUTTON
    # =====================================================
    details_box = [82, 220, 158, 270]

    draw.rounded_rectangle(
        details_box,
        radius=10,
        fill=DARK_PINK
    )

    draw_button_text(
        draw,
        "DETAILS",
        details_box,
        FONT_SMALL,
        WHITE
    )

    # =====================================================
    # BACK BUTTON
    # =====================================================
    back_box = [165, 220, 230, 270]

    draw.rounded_rectangle(
        back_box,
        radius=10,
        fill=BLACK
    )

    draw_button_text(
        draw,
        "BACK",
        back_box,
        FONT_SMALL,
        WHITE
    )

    return img


# =========================================================
# PRODUCT DETAILS SCREEN
# =========================================================
def render_details_screen():

    img = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        PINK_BG
    )

    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [0, 0, WIDTH, 42],
        fill=DARK_PINK
    )

    draw_center_text(
        draw,
        "Product Details",
        11,
        FONT_MEDIUM,
        WHITE
    )

    # =====================================================
    # COPY OF SCANNED PRODUCTS
    # =====================================================
    with state_lock:
        products = list(
            scanned_products.values()
        )

    # =====================================================
    # HEADER
    # =====================================================
    y = 48

    draw.text(
        (5, y),
        "PRODUCT",
        font=FONT_SMALL,
        fill=DARK_PINK
    )

    draw.text(
        (112, y),
        "QTY",
        font=FONT_SMALL,
        fill=DARK_PINK
    )

    draw.text(
        (150, y),
        "PRICE",
        font=FONT_SMALL,
        fill=DARK_PINK
    )

    draw.text(
        (202, y),
        "TOTAL",
        font=FONT_SMALL,
        fill=DARK_PINK
    )

    y += 17

    # =====================================================
    # NO PRODUCT CASE
    # =====================================================
    if not products:

        draw_center_text(
            draw,
            "No products scanned",
            115,
            FONT_SMALL,
            DARK_PINK
        )

    else:

        total_price = 0.0

        # Each product uses two lines:
        # Line 1 = Name / Qty / Price / Total
        # Line 2 = Barcode
        #
        # Maximum visible products are automatically limited
        # so the BACK button and TOTAL remain visible.
        max_visible_products = 5

        visible_products = products[
            :max_visible_products
        ]

        for item in visible_products:

            product_name = item["product"]
            barcode = item["barcode"]
            quantity = item["quantity"]
            unit_price_text = item["unit_price"]

            # ---------------------------------------------
            # Extract numeric unit price
            # Example:
            # "$1.5 per kg" -> 1.5
            # "$0.5 per packet" -> 0.5
            # ---------------------------------------------
            try:

                unit_price = float(
                    unit_price_text
                    .replace("$", "")
                    .split()[0]
                )

            except Exception:

                unit_price = 0.0

            item_total = (
                unit_price * quantity
            )

            total_price += item_total

            # Short product name for screen
            display_name = product_name

            if len(display_name) > 11:
                display_name = (
                    display_name[:10]
                    +
                    "."
                )

            # Price
            price_display = (
                f"${unit_price:.2f}"
            )

            total_display = (
                f"${item_total:.2f}"
            )

            # ---------------------------------------------
            # PRODUCT NAME
            # ---------------------------------------------
            draw.text(
                (5, y),
                display_name,
                font=FONT_SMALL,
                fill=BLACK
            )

            # ---------------------------------------------
            # QUANTITY
            # ---------------------------------------------
            draw.text(
                (115, y),
                str(quantity),
                font=FONT_SMALL,
                fill=BLACK
            )

            # ---------------------------------------------
            # UNIT PRICE
            # ---------------------------------------------
            draw.text(
                (150, y),
                price_display,
                font=FONT_SMALL,
                fill=BLACK
            )

            # ---------------------------------------------
            # ITEM TOTAL
            # ---------------------------------------------
            draw.text(
                (202, y),
                total_display,
                font=FONT_SMALL,
                fill=BLACK
            )

            y += 15

            # ---------------------------------------------
            # BARCODE
            # ---------------------------------------------
            barcode_text = (
                "Barcode: "
                +
                barcode
            )

            draw.text(
                (5, y),
                barcode_text,
                font=FONT_SMALL,
                fill=DARK_PINK
            )

            y += 25

        # =================================================
        # IF MORE THAN 5 PRODUCTS WERE SCANNED
        # =================================================
        if len(products) > max_visible_products:

            draw.text(
                (5, 180),
                f"+ {len(products) - max_visible_products} "
                f"more product(s)",
                font=FONT_SMALL,
                fill=DARK_PINK
            )

            # Calculate total for hidden products too
            for item in products[
                max_visible_products:
            ]:

                unit_price_text = item[
                    "unit_price"
                ]

                try:

                    unit_price = float(
                        unit_price_text
                        .replace("$", "")
                        .split()[0]
                    )

                except Exception:

                    unit_price = 0.0

                total_price += (
                    unit_price
                    *
                    item["quantity"]
                )

        # =================================================
        # GRAND TOTAL
        # =================================================
        draw.line(
            [5, 205, 235, 205],
            fill=DARK_PINK,
            width=2
        )

        draw.text(
            (5, 214),
            "TOTAL PRICE",
            font=FONT_MEDIUM,
            fill=DARK_PINK
        )

        draw.text(
            (150, 216),
            f"${total_price:.2f}",
            font=FONT_MEDIUM,
            fill=BLACK
        )

    # =====================================================
    # BACK BUTTON
    # =====================================================
    details_back_box = [65, 285, 175, 315]

    draw.rounded_rectangle(
        details_back_box,
        radius=8,
        fill=BLACK
    )

    draw_button_text(
        draw,
        "BACK",
        details_back_box,
        FONT_SMALL,
        WHITE
    )

    return img


# =========================================================
# PAYMENT QR GENERATION
# =========================================================
def generate_payment_qr():

    global payment_qr_image
    global payment_qr_data

    payment_qr_data = (
        "MYTROLLEY-PAY-"
        +
        uuid.uuid4().hex[:12].upper()
    )

    log(
        f"[PAYMENT] Generated QR data: "
        f"{payment_qr_data}"
    )

    try:

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=5,
            border=2
        )

        qr.add_data(
            payment_qr_data
        )

        qr.make(
            fit=True
        )

        qr_image = qr.make_image(
            fill_color="black",
            back_color="white"
        ).convert("RGB")

        # Resize to fit the 240x320 display
        qr_image.thumbnail(
            (170, 170),
            Image.Resampling.NEAREST
        )

        payment_qr_image = qr_image

        return True

    except Exception as e:

        log(
            f"[PAYMENT] QR generation failed: {e}"
        )

        payment_qr_image = None

        return False


# =========================================================
# PAYMENT SCREEN
# =========================================================
def render_payment_screen():

    img = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        PINK_BG
    )

    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [0, 0, WIDTH, 42],
        fill=DARK_PINK
    )

    draw_center_text(
        draw,
        "Scan to Pay",
        10,
        FONT_MEDIUM,
        WHITE
    )

    if payment_qr_image is not None:

        qr_x = (
            WIDTH
            -
            payment_qr_image.width
        ) // 2

        qr_y = 55

        img.paste(
            payment_qr_image,
            (qr_x, qr_y)
        )

    else:

        draw_center_text(
            draw,
            "QR ERROR",
            120,
            FONT_MEDIUM,
            DARK_PINK
        )

    draw_center_text(
        draw,
        "Payment QR",
        235,
        FONT_SMALL,
        DARK_PINK
    )

    # =====================================================
    # PAID BUTTON
    # =====================================================
    paid_box = [20, 265, 105, 310]

    draw.rounded_rectangle(
        paid_box,
        radius=10,
        fill=DARK_PINK
    )

    draw_button_text(
        draw,
        "PAID",
        paid_box,
        FONT_MEDIUM,
        WHITE
    )

    # =====================================================
    # BACK BUTTON
    # =====================================================
    payment_back_box = [125, 265, 220, 310]

    draw.rounded_rectangle(
        payment_back_box,
        radius=10,
        fill=BLACK
    )

    draw_button_text(
        draw,
        "BACK",
        payment_back_box,
        FONT_MEDIUM,
        WHITE
    )

    return img


# =========================================================
# THANK YOU SCREEN
# =========================================================
def render_thank_you_screen():

    img = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        PINK_BG
    )

    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [0, 0, WIDTH, 50],
        fill=DARK_PINK
    )

    draw_center_text(
        draw,
        "MyTrolley",
        14,
        FONT_LARGE,
        WHITE
    )

    draw_center_text(
        draw,
        "THANK YOU!",
        105,
        FONT_LARGE,
        DARK_PINK
    )

    draw_center_text(
        draw,
        "Payment Successful",
        145,
        FONT_MEDIUM,
        DARK_PINK
    )

    draw_center_text(
        draw,
        "Please visit again.",
        180,
        FONT_SMALL,
        DARK_PINK
    )

    return img


# =========================================================
# VOICE ASSISTANT PIPELINE
# =========================================================
SHOP_DATA = {

    "store": "Balad Grocery Store",

    "inventory": [

        {
            "product": "Rice",
            "barcode": "1111111111",
            "available_quantity": "10 kg",
            "unit_price": "$1.5 per kg",
            "exact_location": {
                "rack": 2,
                "row": 1,
                "shelf": 2
            }
        },

        {
            "product": "Milk",
            "barcode": "2222222222",
            "available_quantity": "3 bottles",
            "unit_price": "$2 per bottle",
            "exact_location": {
                "rack": 1,
                "row": 2,
                "shelf": 1
            }
        },

        {
            "product": "Cooking Oil",
            "barcode": "3333333333",
            "available_quantity": "5 bottles",
            "unit_price": "$4 per bottle",
            "exact_location": {
                "rack": 2,
                "row": 2,
                "shelf": 1
            }
        },

        {
            "product": "Sugar",
            "barcode": "4444444444",
            "available_quantity": "8 kg",
            "unit_price": "$1 per kg",
            "exact_location": {
                "rack": 2,
                "row": 3,
                "shelf": 1
            }
        },

        {
            "product": "Salt",
            "barcode": "5555555555",
            "available_quantity": "12 packets",
            "unit_price": "$0.5 per packet",
            "exact_location": {
                "rack": 2,
                "row": 3,
                "shelf": 2
            }
        },

        {
            "product": "Noodles",
            "barcode": "6666666666",
            "available_quantity": "10 packets",
            "unit_price": "$0.8 per packet",
            "exact_location": {
                "rack": 4,
                "row": 3,
                "shelf": 1
            }
        },

        {
            "product": "Biscuits",
            "barcode": "7777777777",
            "available_quantity": "5 packets",
            "unit_price": "$1.2 per packet",
            "exact_location": {
                "rack": 3,
                "row": 1,
                "shelf": 2
            }
        },

        {
            "product": "Eggs",
            "barcode": "8888888888",
            "available_quantity": "30 pieces",
            "unit_price": "$0.2 per piece",
            "exact_location": {
                "rack": 4,
                "row": 2,
                "shelf": 1
            }
        },

        {
            "product": "Sample Product",
            "barcode": "9999999999",
            "available_quantity": "0",
            "unit_price": "$0",
            "exact_location": {
                "rack": 0,
                "row": 0,
                "shelf": 0
            }
        },

        {
            "product": "Sample Product 2",
            "barcode": "0000000000",
            "available_quantity": "0",
            "unit_price": "$0",
            "exact_location": {
                "rack": 0,
                "row": 0,
                "shelf": 0
            }
        }
    ]
}


# =========================================================
# NEW: REGISTER A SCANNED BARCODE
# =========================================================
def register_scanned_barcode(barcode):

    # Search barcode in inventory
    matched_item = None

    for item in SHOP_DATA["inventory"]:

        if item["barcode"] == barcode:

            matched_item = item
            break

    # Ignore unknown barcode
    if matched_item is None:

        log(
            f"[BARCODE] Unknown barcode ignored: "
            f"{barcode}"
        )

        return

    # =====================================================
    # GET AVAILABLE QUANTITY
    # =====================================================
    available_quantity_text = (
        matched_item["available_quantity"]
    )

    try:

        available_quantity = int(
            available_quantity_text.split()[0]
        )

    except Exception:

        available_quantity = 0

    with state_lock:

        # =================================================
        # ALREADY SCANNED QUANTITY
        # =================================================
        if barcode in scanned_products:

            current_quantity = scanned_products[
                barcode
            ]["quantity"]

        else:

            current_quantity = 0

        # =================================================
        # DO NOT ALLOW SCANNED QUANTITY > STOCK
        # =================================================
        if current_quantity >= available_quantity:

            log(
                "[BARCODE] Maximum available quantity "
                f"reached for {matched_item['product']} "
                f"({available_quantity}). "
                "Additional scan ignored."
            )

            return

        # =================================================
        # INCREASE QUANTITY
        # =================================================
        if barcode in scanned_products:

            scanned_products[
                barcode
            ]["quantity"] += 1

        else:

            scanned_products[barcode] = {

                "product":
                    matched_item["product"],

                "barcode":
                    matched_item["barcode"],

                "unit_price":
                    matched_item["unit_price"],

                "quantity":
                    1
            }

        quantity = scanned_products[
            barcode
        ]["quantity"]

    log(
        "[SCANNED PRODUCT] "
        f"{matched_item['product']} | "
        f"Barcode: {barcode} | "
        f"Quantity: {quantity}/{available_quantity} | "
        f"Price: {matched_item['unit_price']}"
    )
    show_barcode_modal(barcode, matched_item['product']);


# =========================================================
# COMPLETE PAYMENT
# REDUCE INVENTORY + CLEAR CART + CLEAR CONVERSATION
# =========================================================
def complete_payment():

    global total_products_count
    global counted_ids_set
    global last_detected_barcode
    global last_barcode_time
    global current_aruco_action
    global payment_qr_image
    global payment_qr_data
    global conversation_history
    global current_ui_state
    global thank_you_until

    log(
        "[PAYMENT] Payment confirmed."
    )

    # =====================================================
    # REDUCE INVENTORY BASED ON SCANNED PRODUCTS
    # =====================================================
    with state_lock:

        for barcode, scanned_item in scanned_products.items():

            scanned_quantity = scanned_item["quantity"]

            for inventory_item in SHOP_DATA["inventory"]:

                if inventory_item["barcode"] == barcode:

                    try:

                        current_stock = int(
                            inventory_item[
                                "available_quantity"
                            ].split()[0]
                        )

                    except Exception:

                        current_stock = 0

                    new_stock = max(
                        0,
                        current_stock - scanned_quantity
                    )

                    unit_text = (
                        inventory_item[
                            "available_quantity"
                        ]
                    )

                    parts = unit_text.split()

                    if len(parts) > 1:

                        inventory_item[
                            "available_quantity"
                        ] = (
                            f"{new_stock} "
                            f"{' '.join(parts[1:])}"
                        )

                    else:

                        inventory_item[
                            "available_quantity"
                        ] = str(new_stock)

                    log(
                        "[PAYMENT] Stock updated: "
                        f"{inventory_item['product']} | "
                        f"Purchased: {scanned_quantity} | "
                        f"Remaining: {new_stock}"
                    )

                    break

        # =================================================
        # CLEAR SCANNED PRODUCTS
        # =================================================
        scanned_products.clear()

        # =================================================
        # CLEAR PRODUCT COUNT
        # =================================================
        total_products_count = 0
        counted_ids_set.clear()

        # =================================================
        # CLEAR BARCODE STATE
        # =================================================
        last_detected_barcode = ""
        last_barcode_time = 0

        # =================================================
        # CLEAR ARUCO ACTION STATE
        # =================================================
        current_aruco_action = "STOP"

        # =================================================
        # CLEAR VOICE CONVERSATION / HISTORY
        # =================================================
        conversation_history.clear()

        # =================================================
        # CLEAR PAYMENT QR DATA
        # =================================================
        payment_qr_image = None
        payment_qr_data = ""

        # =================================================
        # SHOW THANK YOU SCREEN
        # =================================================
        thank_you_until = time.time() + 4.0

        current_ui_state = "THANK_YOU"

    log(
        "[PAYMENT] Cart, history, barcode state, "
        "and product count cleared."
    )

    log(
        "[PAYMENT] Thank You screen displayed "
        "for 4 seconds."
    )


SYSTEM_TEMPLATE = """You are Balad, a practical grocery shopkeeper at Balad Grocery Store.

CRITICAL RULES:
1. GREETING: Never greet the customer unless they explicitly say hello or start with a greeting first.
2. INVENTORY ONLY: Answer questions strictly using the INVENTORY TABLE below. Never invent products, quantities, prices, or locations.
3. OUT-OF-STOCK/UNRELATED ITEMS: If a customer asks for an item not in the inventory, or makes a vague comment, politely state that you only sell groceries and ask if they need help finding an item from stock.
4. BREVITY: Keep your replies natural, conversational, and direct between 1 to 2 sentences.
5. FORMATTING: Plain text only. NO emojis, NO bullet points, NO special characters, NO markdown.
6. NO REPETITION: Do not repeat details unless explicitly asked.

INVENTORY TABLE:
{shop_data}

RECENT CONVERSATION:
{conversation}

CUSTOMER SAYS: {customer_question}

SHOPKEEPER REPLY:"""


SAMPLE_RATE = 16000
CHANNELS = 1

AUDIO_FILE = "user_voice.wav"
REPLY_FILE = "reply.wav"

MAX_RECORD_SECONDS = 7
SILENCE_THRESHOLD = 350
SILENCE_DURATION = 1.0
MIN_SPEECH_SECONDS = 0.5

OLLAMA_MODEL = "gemma3:1b"

conversation_history = []


def find_microphone():

    log(
        "Searching for audio input device..."
    )

    try:

        devices = sd.query_devices()

        for i, device in enumerate(devices):

            if (
                device["max_input_channels"] > 0
                and
                "monitor"
                not in device["name"].lower()
            ):

                log(
                    f"Microphone selected: "
                    f"Device {i} - "
                    f"{device['name']}"
                )

                return i

    except Exception as e:

        log(
            f"Microphone search error: {e}"
        )

    return None


MIC_DEVICE = find_microphone()

log("Loading Faster-Whisper Model...")

whisper_model = WhisperModel(
    "tiny.en",
    device="cpu",
    compute_type="int8"
)

log("Loading Pocket TTS Model...")

tts_model = TTSModel.load_model()

voice_state = (
    tts_model.get_state_for_audio_prompt(
        "alba"
    )
)

log("Warming up Ollama Model...")

try:

    chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": "OK"
            }
        ],
        keep_alive="30m"
    )

    log("Ollama ready.")

except Exception as e:

    log(
        f"Ollama Warmup Warning: {e}"
    )


def record_voice():

    global voice_assistant_status

    if MIC_DEVICE is None:

        log(
            "ERROR: Microphone not detected."
        )

        return False

    with state_lock:
        voice_assistant_status = "LISTENING"

    log(
        "[VOICE ASSISTANT] Recording started..."
    )

    start_time = time.time()

    audio_chunks = []

    speech_detected = False
    silence_start = None

    chunk_size = int(
        SAMPLE_RATE * 0.1
    )

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            device=MIC_DEVICE,
            blocksize=chunk_size
        ) as stream:

            while True:

                elapsed = (
                    time.time()
                    -
                    start_time
                )

                if elapsed >= MAX_RECORD_SECONDS:
                    break

                data_chunk, _ = stream.read(
                    chunk_size
                )

                audio_chunks.append(
                    data_chunk.copy()
                )

                volume = np.mean(
                    np.abs(
                        data_chunk.astype(
                            np.int32
                        )
                    )
                )

                if volume > SILENCE_THRESHOLD:

                    if not speech_detected:

                        speech_detected = True

                        log(
                            "[VOICE ASSISTANT] "
                            "Voice detected."
                        )

                    silence_start = None

                else:

                    if speech_detected:

                        if silence_start is None:
                            silence_start = time.time()

                        if (
                            time.time()
                            -
                            silence_start
                            >= SILENCE_DURATION
                            and
                            elapsed
                            >= MIN_SPEECH_SECONDS
                        ):
                            break

        if (
            not audio_chunks
            or
            not speech_detected
        ):

            log(
                "[VOICE ASSISTANT] "
                "No voice detected."
            )

            return False

        audio = np.concatenate(
            audio_chunks,
            axis=0
        )

        wavfile.write(
            AUDIO_FILE,
            SAMPLE_RATE,
            audio
        )

        return True

    except Exception as e:

        log(
            f"[VOICE ASSISTANT ERROR] "
            f"Recording failed: {e}"
        )

        return False


def speech_to_text():

    global voice_assistant_status

    if not os.path.exists(
        AUDIO_FILE
    ):
        return ""

    try:

        with state_lock:
            voice_assistant_status = "THINKING"

        log(
            "[VOICE ASSISTANT] "
            "Transcribing audio with Whisper..."
        )

        segments, _ = (
            whisper_model.transcribe(
                AUDIO_FILE,
                beam_size=5
            )
        )

        text = " ".join(
            s.text
            for s in segments
        ).strip()

        log(
            f"[VOICE ASSISTANT] "
            f"STT Result: '{text}'"
        )

        return text

    except Exception as e:

        log(
            f"[VOICE ASSISTANT ERROR] "
            f"STT Failed: {e}"
        )

        return ""


def generate_ai_response(
    customer_question
):

    global voice_assistant_status

    with state_lock:
        voice_assistant_status = "THINKING"

    log(
        "[VOICE ASSISTANT] "
        "Querying Ollama..."
    )

    conv_str = (
        "\n".join(
            conversation_history[-6:]
        )
        if conversation_history
        else
        "No previous conversation."
    )

    prompt = SYSTEM_TEMPLATE.format(
        shop_data=str(SHOP_DATA),
        conversation=conv_str,
        customer_question=customer_question
    )

    try:

        resp = chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.2,
                "num_predict": 100,
                "repeat_penalty": 1.2
            },
            keep_alive="30m"
        )

        ai_reply = (
            resp.message.content.strip()
        )

        log(
            f"[VOICE ASSISTANT] "
            f"Ollama Reply: '{ai_reply}'"
        )

        conversation_history.append(
            f"Customer: {customer_question}"
        )

        conversation_history.append(
            f"Shopkeeper: {ai_reply}"
        )

        if len(conversation_history) > 8:
            del conversation_history[:-8]

        return ai_reply

    except Exception as e:

        log(
            f"[VOICE ASSISTANT ERROR] "
            f"Ollama Chat Failed: {e}"
        )

        return (
            "I am having trouble "
            "checking stock right now."
        )


def text_to_speech(text):

    try:

        audio = tts_model.generate_audio(
            voice_state,
            text
        )

        wavfile.write(
            REPLY_FILE,
            tts_model.sample_rate,
            audio.numpy()
        )

        return True

    except Exception as e:

        log(
            f"[VOICE ASSISTANT ERROR] "
            f"TTS Synthesis Failed: {e}"
        )

        return False


def play_audio():

    global voice_assistant_status

    if not os.path.exists(
        REPLY_FILE
    ):
        return

    try:

        with state_lock:
            voice_assistant_status = "SPEAKING"

        log(
            "[VOICE ASSISTANT] "
            "Playing TTS audio..."
        )

        os.system(
            f"aplay -q '{REPLY_FILE}'"
        )

    finally:

        with state_lock:
            voice_assistant_status = "READY"


def handle_voice_interaction():

    global voice_assistant_status

    if voice_assistant_status != "READY":

        log(
            "[VOICE ASSISTANT] "
            "Busy, skipping request."
        )

        return

    log(
        "=== VOICE INTERACTION TRIGGERED ==="
    )

    if record_voice():

        text = speech_to_text()

        if text:

            reply = generate_ai_response(
                text
            )

            if text_to_speech(reply):

                play_audio()

                return

    with state_lock:
        voice_assistant_status = "READY"


# =========================================================
# HARDWARE PUSH BUTTON
# =========================================================
log(
    "Setting up Hardware Voice Button "
    "on GPIO 17..."
)

try:

    hw_button = Button(
        BUTTON_PIN,
        pull_up=True,
        bounce_time=1.0
    )

    hw_button.when_pressed = lambda: (
        threading.Thread(
            target=handle_voice_interaction,
            daemon=True
        ).start()
    )

    log(
        "Hardware Button active."
    )

except Exception as e:

    log(
        f"Hardware Button Warning: {e}"
    )


# =========================================================
# CAMERA 0 THREAD
# ADVANCED BARCODE SCANNER + ARUCO FOLLOW
# =========================================================
def scan_barcode_pipeline(
    frame_rgb
):

    gray = cv2.cvtColor(
        frame_rgb,
        cv2.COLOR_RGB2GRAY
    )

    resized = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC
    )

    enhanced = cv2.equalizeHist(
        resized
    )

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY
        +
        cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    for img_variant in [
        gray,
        resized,
        enhanced,
        otsu,
        adaptive
    ]:

        try:

            barcodes = decode(
                img_variant
            )

            if len(barcodes) > 0:
                return barcodes

        except Exception:
            pass

    return []


def run_camera_0():

    global last_detected_barcode
    global last_barcode_time
    global current_aruco_action

    log(
        "Initializing Camera 0 "
        "(Barcode Scanner + ArUco Follow)..."
    )

    # Serial ESP32 connection
    try:

        esp32 = serial.Serial(
            "/dev/ttyUSB0",
            115200,
            timeout=1
        )

        time.sleep(1)

        log(
            "ESP32 Serial connected "
            "on /dev/ttyUSB0."
        )

    except Exception as e:

        log(
            f"Serial Warning: "
            f"ESP32 not connected: {e}"
        )

        esp32 = None

    # Camera Calibration
    try:

        calib = np.load(
            "camera_calibration.npz"
        )

        camera_matrix = calib[
            "camera_matrix"
        ]

        dist_coeffs = calib[
            "dist_coeffs"
        ]

    except Exception:

        camera_matrix = np.array(
            [
                [1000, 0, 640],
                [0, 1000, 360],
                [0, 0, 1]
            ],
            dtype=np.float32
        )

        dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float32
        )

    dictionary = (
        cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
    )

    detector = cv2.aruco.ArucoDetector(
        dictionary,
        cv2.aruco.DetectorParameters()
    )

    marker_size = 0.06
    half = marker_size / 2.0

    object_points = np.array(
        [
            [-half, half, 0],
            [half, half, 0],
            [half, -half, 0],
            [-half, -half, 0]
        ],
        dtype=np.float32
    )

    try:

        picam2_0 = Picamera2(0)

        config = (
            picam2_0
            .create_preview_configuration(
                main={
                    "size": (1280, 720),
                    "format": "RGB888"
                }
            )
        )

        picam2_0.configure(config)

        picam2_0.start()

        time.sleep(2)

        log(
            "Camera 0 active."
        )

    except Exception as e:

        log(
            f"Camera 0 Initialization Failed: {e}"
        )

        return

    frame_count = 0

    last_cmd_sent = ""
    last_cmd_time = 0

    COMMAND_INTERVAL = 0.05

    while True:

        try:

            frame = picam2_0.capture_array()

            if frame is None:

                time.sleep(0.01)

                continue

            frame_count += 1

            now = time.time()

            # Barcode scanning
            if frame_count % 3 == 0:

                barcodes = (
                    scan_barcode_pipeline(
                        frame
                    )
                )

                for b in barcodes:

                    code_data = (
                        b.data.decode(
                            "utf-8",
                            errors="ignore"
                        ).strip()
                    )

                    if code_data:

                        if (
                            code_data
                            !=
                            last_detected_barcode
                        ) or (
                            now
                            -
                            last_barcode_time
                            >
                            2.0
                        ):

                            last_detected_barcode = (
                                code_data
                            )

                            last_barcode_time = (
                                now
                            )

                            # =================================
                            # REGISTER SCANNED PRODUCT
                            # =================================
                            register_scanned_barcode(
                                code_data
                            )

                            log(
                                "=========================================="
                            )

                            log(
                                "[CAM0 BARCODE DETECTED] "
                                f"Code: {code_data} | "
                                f"Type: {b.type}"
                            )

                            log(
                                "=========================================="
                            )

            # ArUco tracking
            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2GRAY
            )

            corners, ids, _ = (
                detector.detectMarkers(gray)
            )

            action = "STOP"

            if (
                ids is not None
                and
                0 in ids
            ):

                idx = np.where(
                    ids == 0
                )[0][0]

                img_pts = (
                    corners[idx][0]
                    .astype(np.float32)
                )

                cx = int(
                    np.mean(
                        img_pts[:, 0]
                    )
                )

                success, _, tvec = (
                    cv2.solvePnP(
                        object_points,
                        img_pts,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                )

                if success:

                    x = float(
                        tvec[0][0]
                    )

                    y = float(
                        tvec[1][0]
                    )

                    z = float(
                        tvec[2][0]
                    )

                    distance = float(
                        np.sqrt(
                            x * x
                            +
                            y * y
                            +
                            z * z
                        )
                    )

                    if distance <= 0.50:

                        action = "STOP"

                    elif cx < 550:

                        action = "LEFT"

                    elif cx > 730:

                        action = "RIGHT"

                    else:

                        action = "FORWARD"

            current_aruco_action = action

            # ESP32 commands
            if esp32 and esp32.is_open:

                if (
                    action != last_cmd_sent
                    or
                    now - last_cmd_time
                    >= COMMAND_INTERVAL
                ):

                    esp32.write(
                        f"{action}\n".encode()
                    )

                    esp32.flush()

                    if action != last_cmd_sent:

                        log(
                            ">>> "
                            "[CAM0 ESP32 COMMAND CHANGED]: "
                            f"{action}"
                        )

                        last_cmd_sent = action

                    last_cmd_time = now

            time.sleep(0.01)

        except Exception as e:

            log(
                f"Cam0 Loop Error: {e}"
            )

            time.sleep(0.1)


# =========================================================
# CAMERA 1 THREAD
# YOLO INSTANCE SEGMENTATION COUNTER
# =========================================================

def run_camera_1():
    global total_products_count, belt_running, belt_on_streak, belt_off_streak

    log("Initializing USB Camera 1 (YOLO Product Counter)...")

    cap = None
    try:
        model = YOLOWorld("yolov8s-world.pt")

        # ১. কাস্টম আইটেম লিস্ট
        custom_items = [
            "chips packet",
            "biscuit packet",
            "milk carton",
            "juice box",
            "chocolate bar",
            "soap",
            "shampoo bottle",
            "water bottle",
            "can"
        ]
        model.set_classes(custom_items)

        # ------------------ [A4Tech USB ক্যামেরা সেটআপ] ------------------
        # /dev/video1 এর জায়গায় /dev/video0 ব্যবহার করুন
        CAM_PATH = "/dev/video0"
        cap = cv2.VideoCapture(CAM_PATH, cv2.CAP_V4L2)

        # অথবা সরাসরি Integer Index 0 দিতে পারেন (সবচেয়ে নির্ভরযোগ্য):
        # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        # A4Tech ক্যামেরার জন্য MJPG ফরম্যাট এবং রেজুলেশন 640x480 সেট করা
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            log("Camera 1 Initialization Failed: Could not open USB Camera.")
            return

        log("USB Camera 1 active (A4Tech).")

    except Exception as e:
        log(f"Camera 1 Initialization Failed: {e}")
        if cap and cap.isOpened():
            cap.release()
        return

    # ------------------ [মেন লুপ] ------------------
    try:
        while True:
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                results = model.predict(
                    source=frame,
                    conf=0.15,
                    imgsz=320,
                    verbose=False,
                    device="cpu"
                )

                count = len(results[0].boxes) if (len(results) > 0 and results[0].boxes is not None) else 0
                total_products_count = count
                # log(f"DEBUG count={count} relay_active={relay.is_active}")

                # Debouncing logic: টানা ৩টি ফ্রেমে ফলাফল একই থাকলে স্টেট পরিবর্তন হবে
                want_on = (count == 1)

                if want_on:
                    belt_on_streak += 1
                    belt_off_streak = 0
                else:
                    belt_off_streak += 1
                    belt_on_streak = 0

                # ON: 1 ta item, ar belt ager theke off thakle
                if want_on and not belt_running and belt_on_streak >= BELT_DEBOUNCE_FRAMES:
                    log("Detected: 1 Item -> Starting Belt")
                    start_belt()
                    belt_running = True

                # OFF: 0 ba 2+ item, ar belt ager theke on thakle
                elif (not want_on) and belt_running and belt_off_streak >= BELT_DEBOUNCE_FRAMES:
                    log(f"Detected: {count} Items -> Stopping Belt")
                    stop_belt()
                    belt_running = False

                time.sleep(0.05)

            except Exception as e:
                log(f"Cam1 Loop Error: {e}")
                time.sleep(0.1)
    finally:
        if cap and cap.isOpened():
            cap.release()
            
            
# =========================================================
# PERIODIC SYSTEM MONITOR
# =========================================================
def run_status_monitor():

    while True:

        time.sleep(4.0)

        log(
            f"[HEARTBEAT] "
            f"UI State: {current_ui_state} | "
            f"Voice: {voice_assistant_status} | "
            f"Products Counted: "
            f"{total_products_count} | "
            f"Barcode: "
            f"'{last_detected_barcode}' | "
            f"ArUco Action: "
            f"{current_aruco_action}"
        )


# =========================================================
# MAIN GUI & APPLICATION ENTRY POINT
# =========================================================
def main():

    global current_ui_state

    log(
        "Starting MyTrolley Main System..."
    )

    init_display()

    # Boot Screen
    boot_screen()

    # Startup Greeting
    try:

        log(
            "Synthesizing Startup Greeting..."
        )

        if text_to_speech(
            "Good Morning, Customer. "
            "How can I help you? "
            "Are you searching for something specific?"
        ):

            play_audio()

    except Exception as e:

        log(
            f"Startup Greeting Warning: {e}"
        )

    # Background Threads
    threading.Thread(
        target=run_camera_0,
        daemon=True
    ).start()

    threading.Thread(
        target=run_camera_1,
        daemon=True
    ).start()

    threading.Thread(
        target=run_status_monitor,
        daemon=True
    ).start()
    
    
    log(
        "Entering Screen & Touch Event Loop..."
    )

    last_touch_time = 0
    anim_frame = 0

    while True:

        try:

            # =================================================
            # RENDER CURRENT SCREEN
            # =================================================
            if current_ui_state == "MAIN":

                img = render_main_menu()

            elif current_ui_state == "SPEAK":

                img = render_speak_screen(
                    status=voice_assistant_status,
                    frame_num=anim_frame
                )

                anim_frame += 1

            elif current_ui_state == "CHECKOUT":

                img = render_checkout_screen()

            elif current_ui_state == "DETAILS":

                img = render_details_screen()

            elif current_ui_state == "PAYMENT":

                img = render_payment_screen()

            elif current_ui_state == "THANK_YOU":

                img = render_thank_you_screen()

                # Automatically return to MAIN
                if time.time() >= thank_you_until:

                    current_ui_state = "MAIN"

            else:

                current_ui_state = "MAIN"

                img = render_main_menu()

            show_image(img)

            # =================================================
            # TOUCH INPUT
            # =================================================
            pt = read_touch()

            current_time = time.time()

            if (
                pt
                and
                (
                    current_time
                    -
                    last_touch_time
                    >
                    0.35
                )
            ):

                x, y = pt

                last_touch_time = current_time

                log(
                    f"Touch: x={x}, y={y}"
                )

                # =================================================
                # MAIN SCREEN
                # =================================================
                if current_ui_state == "MAIN":

                    if (
                        30 <= x <= 210
                        and
                        90 <= y <= 160
                    ):

                        log(
                            "UI -> "
                            "Navigated to SPEAK Screen"
                        )

                        current_ui_state = "SPEAK"

                    elif (
                        30 <= x <= 210
                        and
                        190 <= y <= 260
                    ):

                        log(
                            "UI -> "
                            "Navigated to CHECKOUT Screen"
                        )

                        current_ui_state = "CHECKOUT"

                # =================================================
                # SPEAK SCREEN
                # =================================================
                elif current_ui_state == "SPEAK":

                    if (
                        20 <= x <= 115
                        and
                        215 <= y <= 260
                    ):

                        log(
                            "UI -> "
                            "Touch 'TALK' button pressed"
                        )

                        threading.Thread(
                            target=handle_voice_interaction,
                            daemon=True
                        ).start()

                    elif (
                        125 <= x <= 220
                        and
                        215 <= y <= 260
                    ):

                        log(
                            "UI -> "
                            "Returned to MAIN Screen"
                        )

                        current_ui_state = "MAIN"

                # =================================================
                # CHECKOUT SCREEN
                # =================================================
                elif current_ui_state == "CHECKOUT":

                    # PAY
                    if (
                        10 <= x <= 75
                        and
                        220 <= y <= 270
                    ):

                        log(
                            "UI -> "
                            "PAY button pressed"
                        )

                        if generate_payment_qr():

                            current_ui_state = "PAYMENT"

                        else:

                            log(
                                "UI -> "
                                "QR generation failed"
                            )

                    # VIEW DETAILS
                    elif (
                        82 <= x <= 158
                        and
                        220 <= y <= 270
                    ):

                        log(
                            "UI -> "
                            "VIEW DETAILS pressed"
                        )

                        current_ui_state = "DETAILS"

                    # BACK
                    elif (
                        165 <= x <= 230
                        and
                        220 <= y <= 270
                    ):

                        log(
                            "UI -> "
                            "Checkout BACK pressed"
                        )

                        current_ui_state = "MAIN"

                # =================================================
                # DETAILS SCREEN
                # =================================================
                elif current_ui_state == "DETAILS":

                    if (
                        65 <= x <= 175
                        and
                        285 <= y <= 315
                    ):

                        log(
                            "UI -> "
                            "Details BACK pressed"
                        )

                        current_ui_state = "CHECKOUT"

                # =================================================
                # PAYMENT SCREEN
                # =================================================
                elif current_ui_state == "PAYMENT":

                    # =================================================
                    # PAID
                    # =================================================
                    if (
                        20 <= x <= 105
                        and
                        265 <= y <= 310
                    ):

                        log("UI -> PAID button pressed")

                        # ১. পেমেন্ট কমপ্লিট করা
                        complete_payment()
                        
                        # ২. দরজা খোলার আগেই থ্যাংক ইউ স্ক্রিন ডিসপ্লেতে ফোর্স রেন্ডার করা
                        img = render_thank_you_screen()
                        show_image(img)

                        # ৩. দরজা খোলা ও বন্ধ করা
                        open_doors()
                        sleep(5)
                        close_doors()

                        # ৪. দরজা বন্ধ হওয়ার পর থেকে ৩ সেকেন্ড থ্যাংক ইউ স্ক্রিন ধরে রাখার জন্য టైমার আপডেট করা
                        thank_you_until = time.time() + 3.0
                        
                    # =================================================
                    # BACK
                    # =================================================
                    elif (
                        125 <= x <= 220
                        and
                        265 <= y <= 310
                    ):

                        log(
                            "UI -> "
                            "Payment BACK pressed"
                        )

                        current_ui_state = "CHECKOUT"

            time.sleep(0.04)

        except KeyboardInterrupt:

            log(
                "Shutting down Application..."
            )

            break

        except Exception as e:

            log(
                f"UI Loop Exception: {e}"
            )

            time.sleep(0.1)


# =========================================================
# CLEANUP HANDLER
# =========================================================
def cleanup_resources(
    signum=None,
    frame=None
):

    log(
        "Cleaning up GPIO, SPI, "
        "and Audio handles..."
    )

    try:
        sd.stop()
    except Exception:
        pass

    try:

        lcd_spi.close()
        touch_spi.close()

        lgpio.gpio_free(
            gpio_chip,
            DC_PIN
        )

        lgpio.gpio_free(
            gpio_chip,
            RESET_PIN
        )

        lgpio.gpiochip_close(
            gpio_chip
        )

    except Exception:
        pass
        
    try:
        relay.off()
    except Exception:
        pass

    os._exit(0)


signal.signal(
    signal.SIGINT,
    cleanup_resources
)

signal.signal(
    signal.SIGTERM,
    cleanup_resources
)


# =========================================================
# PROGRAM START
# =========================================================
if __name__ == "__main__":

    try:

        main()

    finally:

        cleanup_resources()


