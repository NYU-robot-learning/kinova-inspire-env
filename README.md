# Kinova Inspire Environment

Gymnasium-compatible environment for teleoperating a Kinova J2N6S300 arm with an Inspire RH56 hand, using Aria camera for visual feedback and hand tracking.

## Quick Start

```python
from kinova_inspire_env import KinovaInspireEnv
import numpy as np

env = KinovaInspireEnv(
    aria_ip="100.94.225.27",
    aria_port=10114,
    hand_port="/dev/ttyUSB0"
)

obs = env.reset(init=True)
action = np.zeros(31)  # 16 (wrist pose) + 15 (fingertip positions)
obs, reward, done, truncated, info = env.step(action)
```

## Installation

1. **Clone and initialize submodules:**
```bash
git clone <repository-url>
cd kinova-inspire-env
git submodule update --init --recursive
```

2. **Install dependencies:**
```bash
pip install -r kinova_calibration/requirements.txt
pip install gymnasium ikpy pyserial
```

3. **Calibration:** Ensure `kinova_calibration/data/eye_to_base.npy` exists (see `kinova_calibration/README.md`)

## Configuration

Edit `kinova_calibration/config.py`:

```python
ARIA_IP = "100.94.225.27"
ARIA_PORT = 10114
KINOVA_IP = "100.96.31.124"
RIGHT_HAND_PORT = "/dev/ttyUSB0"  # Adjust for your system
```

## Usage

### Action Format

Action is a 31-dimensional vector:
- **First 16 values**: Wrist pose (4×4 homogeneous matrix, flattened) in device frame
- **Last 15 values**: Fingertip positions (5 fingers × 3D) in device frame
  - Finger order: index, middle, ring, pinky, thumb

The environment automatically transforms coordinates: device → camera → base frame.

### Recording Hand Data

```bash
python record_grasp.py
```

Saves to `data/`: wrist pose and fingertip positions from Aria.

### Using Recorded Data

```python
from utils.transfrom_utils import wrist_to_homogeneous
import numpy as np

# Load recorded data
wrist_axes = np.load("data/wrist_axes_device.npy", allow_pickle=True).item()
wrist_translation = np.load("data/wrist_translation_device.npy")
fingertips = np.load("data/fingertips_device.npy")

# Convert to action
wrist_pose = wrist_to_homogeneous(wrist_translation, wrist_axes).reshape(16)
action = np.concatenate([wrist_pose, fingertips.reshape(15)])

env.step(action)
```

## Project Structure

```
kinova-inspire-env/
├── kinova_inspire_env.py      # Main Gymnasium environment
├── record_grasp.py            # Record hand data from Aria
├── data/                      # Recorded hand data
├── inspire_sdk/               # Hand SDK (submodule)
├── kinova_calibration/        # Calibration tools (submodule)
│   └── data/eye_to_base.npy # Required calibration file
└── utils/
    ├── constant.py           # Kinova chain definition
    └── transfrom_utils.py    # Coordinate transformations
```

## API Reference

### `KinovaInspireEnv`

- `__init__(aria_ip, aria_port, hand_port, enable_hand_control=True, enable_kinova_control=True, forces=40, speeds=100)`
- `reset(init=True)` → Returns RGB image observation
- `step(action)` → Returns (obs, reward, terminated, truncated, info)
- `get_states()` → Returns current RGB image

### `InspireHandOperator` (from `inspire_sdk`)

- `step(fingertip_positions)` → Controls hand using 5×3 fingertip targets
- `reset()` → Resets hand to default pose

## Troubleshooting

| Issue                  | Solution                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Hand not connecting    | Check serial port: `ls /dev/ttyUSB*` (Linux) or `/dev/tty.usbserial*` (macOS). Update `RIGHT_HAND_PORT` in config. |
| Aria connection failed | Verify streaming server is running. Check network to `ARIA_IP:ARIA_PORT`.                                          |
| Calibration error      | Ensure `kinova_calibration/data/eye_to_base.npy` exists. Run calibration (see `kinova_calibration/README.md`).     |
| Arm control issues     | Verify Kinova is powered and Kinova publisher is running. Check network to `KINOVA_IP`.                            |

## Dependencies

- `gymnasium`, `numpy`, `opencv-python`, `scipy`, `mujoco`, `pyzmq`, `ikpy`, `pyserial`
