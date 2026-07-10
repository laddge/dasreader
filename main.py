import cv2
import numpy as np
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
import socket
import struct

TARGET_IP = "192.168.0.255"
TARGET_PORT = 6454
UNIVERSE = 0

header = b"Art-Net\x00"
opcode = struct.pack("<H", 0x5000)
prot_ver = struct.pack(">H", 14)
sequence = b"\x00"
physical = b"\x00"
universe = struct.pack("<H", UNIVERSE)
length = struct.pack(">H", 512)
packet = header + opcode + prot_ver + sequence + physical + universe + length
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

t512 = cv2.imread("t512.png")
t256 = cv2.cvtColor(cv2.imread("t256.png"), cv2.COLOR_BGR2GRAY)
fg = np.array([162, 108, 55])
mask = cv2.inRange(t512, fg, fg)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
boxes = np.array([cv2.boundingRect(c) for c in contours])
indices = np.lexsort((boxes[:, 0], boxes[:, 1]))
boxes = boxes[indices]
template = np.array([t256[y:y+h, x:x+w].flatten() for x, y, w, h in boxes[:256]])
A_norm = template / np.linalg.norm(template, axis=1, keepdims=True)

capture = WindowsCapture(
    cursor_capture=False,
    window_name="DMX LEVELS",
)

@capture.event
def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):

    img = cv2.cvtColor(frame.convert_to_bgr().frame_buffer, cv2.COLOR_BGR2GRAY)
    target = np.array([img[y:y+h, x:x+w].flatten() for x, y, w, h in boxes])
    B_norm = target / np.linalg.norm(target, axis=1, keepdims=True)
    similarity_matrix = B_norm @ A_norm.T
    data = np.argmax(similarity_matrix, axis=1).astype(np.uint8)
    sock.sendto(packet+bytes(data), (TARGET_IP, TARGET_PORT))
    print(data)

@capture.event
def on_closed():
    print("Capture session closed")

capture.start()
