import gymnasium as gym
from gymnasium import spaces
import numpy as np
import base64
import cv2
import time
from pathlib import Path

from kinova_calibration.utils.aria_sub import AriaSubscriber
from kinova_calibration.config import *
from utils.transfrom_utils import *
from utils.constant import *
from inspire_sdk.hand_operator import InspireHandOperator
from kinova_calibration.utils.robot.kinova import KinovaArm

class KinovaInspireEnv(gym.Env):
    def __init__(
        self,
        aria_ip=ARIA_IP,
        aria_port=ARIA_PORT,
        hand_port=RIGHT_HAND_PORT,
        enable_hand_control=True, 
        enable_kinova_control=True,
        forces=40,
        speeds=100,
    ):
        self.aria_subscriber = AriaSubscriber(aria_ip, aria_port)
        self.enable_hand_control = enable_hand_control
        self.enable_kinova_control = enable_kinova_control

        if enable_hand_control:
             self.hand_operator = InspireHandOperator(port=hand_port, forces=[forces] * 6, speeds=[speeds] * 6)
        if enable_kinova_control:
            self.arm_operator = KinovaArm("left")
        
        kinova_calib_dir = Path(__file__).parent / "kinova_calibration"
        camera_to_base_path = kinova_calib_dir / "data" / "eye_to_base.npy"
        self.T_Camera_Base = np.load(camera_to_base_path)

        self.T_Device_Camera = None
        
        while self.T_Device_Camera is None:
            message = self.aria_subscriber.recv_message()
            if message is None:
                continue
            
            calibration = message.get("calibration")
            if calibration is not None:
                if "T_Device_Camera" in calibration:
                    self.T_Device_Camera = np.array(calibration["T_Device_Camera"])
                    print(f"Loaded T_Device_Camera: {self.T_Device_Camera}")
                
                # Extract image dimensions from calibration if available
                if "image_width" in calibration:
                    image_width = calibration["image_width"]
                if "image_height" in calibration:
                    image_height = calibration["image_height"]
                
                if self.T_Device_Camera is not None:
                    break
        
        if self.T_Device_Camera is None:
            raise RuntimeError("Could not load T_Device_Camera from messages")
        
        print(f"Image size: {image_width} x {image_height}")

        self.observation_space = spaces.Box(
            low=0, 
            high=255,
            shape=(image_height, image_width, 3),
            dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(31,),  # 4 * 4 + 5 * 3 = 31
            dtype=np.float32
        )
        
    def _decode_rgb_image(self, rgb_base64):
        if isinstance(rgb_base64, str):
            jpeg_bytes = base64.b64decode(rgb_base64)
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            bgr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            return rgb_image
        return rgb_base64

    def get_states(self):
        message = self.aria_subscriber.recv_message()
        if message.get("rgb") is not None:
            self.current_image = self._decode_rgb_image(message.get("rgb"))
            return self.current_image
        return None

    def reset(self, init=True):
        if self.enable_kinova_control:
            self.arm_operator.move_joint(kinova_reset_joint_angles)
        
        if self.enable_hand_control:
            if init:
                self.hand_operator.reset()
        
        return self.get_states()
        

    def step(self, action):
        # Action is the wrist pose in device frame (4, 4) + 5 fingertip position in device frame
        wrist_pose = action[:16].reshape(4, 4)
        wrist_frame = wrist_pose.copy()
        fingertip_positions = action[16:].reshape(5, 3)
        fingertip_projected = []

        for fingertip in fingertip_positions:
            pos = project_point_to_axis(fingertip, wrist_frame)
            pos = rotate_point_around_axis(pos, np.array([0, 0, 1]), -90)
            pos[1] -= 0.03
            fingertip_projected.append(pos)
        fingertip_projected = np.array(fingertip_projected)

        wrist_pose = transform_device_to_camera(wrist_pose, self.T_Device_Camera)
        wrist_pose = transform_camera_to_base(wrist_pose, self.T_Camera_Base)
        
        kinova_eef = axis_local_transform(wrist_pose, "x", 0.13)
        eef = axis_local_rotate(kinova_eef, 'z', -90)
        eef = axis_local_rotate(eef, 'x', -90)
        
        target_wrist_position = eef[:3, 3]
        target_wrist_orientation = eef[:3, :3]
        ja = kinova_j2n6s300_chain.inverse_kinematics(
            target_position=target_wrist_position,
            target_orientation=target_wrist_orientation,
            orientation_mode="all", 
            initial_position=kinova_initial_position
        )
        joint_angles = ja[1:-1]
        joint_angles[-1] += np.deg2rad(30)
        if self.enable_kinova_control:
            self.arm_operator.move_joint(joint_angles)
        
        time.sleep(10.0)

        if self.enable_hand_control:
            self.hand_operator.step(fingertip_projected)

        state = self.get_states()
        return state, 0.0, False, False, {}


if __name__ == "__main__":
    env = KinovaInspireEnv()
    env.reset(init=True)
 

    example_wrist_axes = np.load("data/wrist_axes_device.npy", allow_pickle=True).item()
    example_wrist_translation = np.load("data/wrist_translation_device.npy", allow_pickle=True)
    example_fingertips = np.load("data/fingertips_device.npy", allow_pickle=True)

    example_wrist_pose = wrist_to_homogeneous(example_wrist_translation, example_wrist_axes)
    example_wrist_pose = example_wrist_pose.reshape(16, 1)
    example_fingertips = example_fingertips.reshape(15, 1)
    example_action = np.concatenate([example_wrist_pose, example_fingertips], axis=0)

    env.step(example_action)
    time.sleep(1)
    env.reset(init=False)


