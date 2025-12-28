from kinova_calibration.utils.aria_sub import AriaSubscriber
from kinova_calibration.config import *
import numpy as np
import cv2
import base64
import pickle
import zmq
from pathlib import Path
from utils.transfrom_utils import *

FINGERTIP_INDICES = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'pinky': 4}
FINGER_ORDER = ['index', 'middle', 'ring', 'pinky', 'thumb']

def _decode_rgb_image(rgb_base64):
    if isinstance(rgb_base64, str):
        jpeg_bytes = base64.b64decode(rgb_base64)
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image
    return rgb_base64


if __name__ == "__main__":
    aria_sub = AriaSubscriber(ARIA_IP, ARIA_PORT)

    input("Press Enter to record grasp...")

    # Try to receive message non-blocking
    message = aria_sub.recv_message()
    if message is None:
        print("no hand")
        exit(0)
    
    right_hand = message.get("right_hand")
    if right_hand is None:
        print("no hand")
        exit(0)
    
    hand_data = right_hand
    print("Found right hand data")

    wrist_translation_device = np.array(hand_data.get("wrist_translation"))
    wrist_axes_device = hand_data.get("wrist_axes")
    landmarks_device = hand_data.get("landmarks_device")
    
    if landmarks_device is not None:
        landmarks_device = np.array(landmarks_device)
        fingertips_device = [landmarks_device[FINGERTIP_INDICES[finger]] for finger in FINGER_ORDER]
        print(f"Fingertips shape: {np.array(fingertips_device).shape}")
        print(f"Landmarks shape: {landmarks_device.shape}")
        np.save("data/landmarks_device.npy", landmarks_device)
        np.save("data/fingertips_device.npy", np.array(fingertips_device))
    else:
        print("Warning: No landmarks found")
    
    print(f"Wrist translation shape: {wrist_translation_device.shape}")
    print(f"Wrist axes: {wrist_axes_device}")
    np.save("data/wrist_translation_device.npy", wrist_translation_device)
    np.save("data/wrist_axes_device.npy", wrist_axes_device)