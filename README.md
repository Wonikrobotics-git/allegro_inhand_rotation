# allegro_inhand_rotation
Reference Implementation of In-Hand Object Rotation for Allegro Hand Platforms

This repository provides an implementation example for in-hand object rotation using the **Allegro Hand Platforms**.
It combines a ROS2-based hardware controller with an AI-driven manipulation algorithm originally developed for in-hand rotation research.

This implementation currently supports **Allegro Hand V4**, but the software architecture is designed to be modular and extendable.
As additional robotic hand platforms are developed within our organization, this codebase may be expanded to include **plug-in modules and adapters** for new hardware versions, enabling broader compatibility across future Wonik Robotics hand systems.

This codebase utilizes:

- **Allegro Hand ROS2 Controller**
  https://github.com/Wonikrobotics-git/allegro_hand_ros2
- **AI Algorithm: In-Hand Object Rotation via Rapid Motor Adaptation (RMA)**
  Original research & implementation by Haozhi Qi
  https://haozhi.io/hora/


## Test System Configuration

- Ubuntu 22.04
- ROS2 Humble
- Allegro Hand V4 (Right side)
- IsaacGym 4.0 Simulator


### System Requirements

1. Isaac Gym 4.0 Installation

Download: [https://developer.nvidia.com/isaac-gym/download](https://developer.nvidia.com/isaac-gym/download)

**Prerequisites**

```
• Ubuntu 18.04 or 20.04
• Python 3.6, 3.7, or 3.8
• NVIDIA driver version 470.74 or newer
• GPU: NVIDIA Pascal or later, with at least 8 GB VRAM
```

2. Allegro Hand Controller (ROS 2 Humble)

The **ROS 2 Humble Python API requires Python 3.10+** to run the Allegro Hand controller.

```
• ROS 2 Humble
• Python 3.10 or later
```

⚠️ Important Notice

Isaac Gym and ROS 2 Humble have **incompatible Python version requirements**:

* **Training Phase (Isaac Gym):** Python 3.6–3.8
* **Inference Phase (ROS 2 Humble):** Python 3.10+

To avoid version conflicts and ensure smooth execution of both phases,
**we strongly recommend using separate Conda virtual environments**:

* One environment dedicated to Isaac Gym training
* Another environment dedicated to ROS 2 inference (Allegro Hand controller)

This separation prevents dependency collisions and ensures stable operation across the full pipeline.

### Conda Setting

```
# for algorithm training
conda create -n hora python=3.8

# for algorithm deploying
conda create -n allegro python=3.10
```

### Check each conda environments

- `hora` env check: `python compare_hands.py`
  - This will show you the difference between AllegroHandHora urdf and PublicAllegroHandHora urdf

<center>
<img src="./matarials/compare.gif" width="50%" />
</center>

- `allegro` env check: `python hora/algo/allegro_ros2.py`
  - Command Interface script

```

```

## Run

### Generate Grasping Poses

To achieve a stable initial grasp, you must prepare reliable grasp poses for the target objects.
According to the [original HORA instructions](https://github.com/HaozhiQi/hora/?tab=readme-ov-file#prerequisite), you can directly download the provided `.npy` grasp pose files for:

* **Public Allegro Hand**
* **HORA internal Allegro Hand** (features slightly longer fingertips than the standard Allegro Hand V4)

We strongly recommend reviewing the original instructions to understand the differences and verify the available data files.

Alternatively, you can generate grasp poses **from scratch** using the scripts included in this repository:

```bash
./scripts/gen_grasp.sh
```

This script will run the full grasp-pose generation pipeline and produce the necessary `.npy` files for training or evaluation.


### Train

Major Difference: Use All Shape Variations (Ball, Cyclinder, Cude, ... Etc.)

- Stage 1: Use privileged observation
- Stage 2: Only use proprioceptive observation(history)

### Test in Simulation

- Evaluation
- Visualization Each Stage to check training result

### Test in Real-world

1. Start Allegro (Right) Hand

```
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

```
ros2 launch allegro_hand_bringup allegro_hand.launch.py
```

2. Run Hora Algorithm

```
scripts/deploy.sh
```


---

## License

This repository is licensed under the MIT License.

- Modifications and integration by [**WonikRobotics_official**](https://github.com/Wonikrobotics-git) (© 2025)
- Additional contributions by [**Jungyeon Lee**](https://github.com/curieuxjy) (© 2025)

The full license text is available in the [LICENSE](./LICENSE) file.
