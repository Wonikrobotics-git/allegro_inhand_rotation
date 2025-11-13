# allegro_inhand_rotation
Implementation Example of In-Hand Object Rotation using the Allegro Hand Platforms

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

---

## License

This repository is licensed under the MIT License.

- Original codebase (RMA implementation) by [**Haozhi Qi**](https://github.com/HaozhiQi) (© 2022)  
- Modifications and integration by [**WonikRobotics_official**](https://github.com/Wonikrobotics-git) (© 2025)  
- Additional contributions by [**Jungyeon Lee**](https://github.com/curieuxjy) (© 2025)

The full license text is available in the [LICENSE](./LICENSE) file.
