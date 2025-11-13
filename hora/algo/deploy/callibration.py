import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState


class Allegro(object):

    def __init__(self, hand_topic_prefix="allegroHand_0", num_joints=16):
        """Simple python interface to the Allegro Hand."""
        hand_topic_prefix = hand_topic_prefix.rstrip("/")
        topic_grasp_command = "{}/lib_cmd".format(hand_topic_prefix)
        topic_joint_command = "{}/joint_cmd".format(hand_topic_prefix)
        self.pub_grasp = rospy.Publisher(topic_grasp_command, String, queue_size=10)
        self.pub_joint = rospy.Publisher(topic_joint_command, JointState, queue_size=10)
        topic_joint_state = "{}/joint_states".format(hand_topic_prefix)
        rospy.Subscriber(topic_joint_state, JointState, self._joint_state_callback)
        self._joint_state = None
        self._num_joints = num_joints
        rospy.loginfo("Allegro Client start with hand topic: {}".format(hand_topic_prefix))
        self._named_grasps_mappings = {
            "home": "home",
            "ready": "ready",
            "three_finger_grasp": "grasp_3",
            "three finger grasp": "grasp_3",
            "four_finger_grasp": "grasp_4",
            "four finger grasp": "grasp_4",
            "index_pinch": "pinch_it",
            "index pinch": "pinch_it",
            "middle_pinch": "pinch_mt",
            "middle pinch": "pinch_mt",
            "envelop": "envelop",
            "off": "off",
            "gravity_compensation": "gravcomp",
            "gravity compensation": "gravcomp",
            "gravity": "gravcomp",
        }

    def disconnect(self):
        self.command_hand_configuration("off")

    def _joint_state_callback(self, data):
        self._joint_state = data

    def command_joint_position(self, desired_pose):
        if not hasattr(desired_pose, "__len__") or len(desired_pose) != self._num_joints:
            rospy.logwarn("Desired pose must be a {}-d array: got {}.".format(self._num_joints, desired_pose))
            return False
        msg = JointState()
        try:
            msg.position = desired_pose
            self.pub_joint.publish(msg)
            return True
        except rospy.exceptions.ROSSerializationException:
            rospy.logwarn("Incorrect type for desired pose: {}.".format(desired_pose))
            return False

    def command_joint_torques(self, desired_torques):
        if not hasattr(desired_torques, "__len__") or len(desired_torques) != self._num_joints:
            rospy.logwarn("Desired torques must be a {}-d array: got {}.".format(self._num_joints, desired_torques))
            return False
        msg = JointState()
        try:
            msg.effort = desired_torques
            self.pub_joint.publish(msg)
            rospy.logdebug("Published desired torques.")
            return True
        except rospy.exceptions.ROSSerializationException:
            rospy.logwarn("Incorrect type for desired torques: {}.".format(desired_torques))
            return False

    def poll_joint_position(self, wait=False):
        if wait:
            self._joint_state = None
            while not self._joint_state:
                rospy.sleep(0.001)
        if self._joint_state:
            return (self._joint_state.position, self._joint_state.effort)
        else:
            return None

    def command_hand_configuration(self, hand_config):
        if hand_config in self._named_grasps_mappings:
            msg = String(self._named_grasps_mappings[hand_config])
            rospy.logdebug("Commanding grasp: {}".format(msg.data))
            self.pub_grasp.publish(msg)
            return True
        else:
            rospy.logwarn("Unable to command unknown grasp {}".format(hand_config))
            return False

    def list_hand_configurations(self):
        return self._named_grasps_mappings.keys()
    

class HardwarePlayerCalib(object):
    def __init__(self):
        # self.action_scale = 1 / 24
        self.actions_num = 16
        self.device = "cuda"

        # Initial hand poses (joint positions)
        self.init_pose = [
            0.0627, 1.2923, 0.3383, 0.1088,
            0.0724, 1.1983, 0.1551, 0.1499,
            0.1343, 1.1736, 0.5355, 0.2164,
            1.1202, 1.1374, 0.8535, -0.0852,
        ]
        self.home_pose = [
            0.018135520640395555, -0.25718396127157894, 0.7298018013651694, 0.7611417453908358,
            0.049130370781299726, -0.25031131767222525, 0.739642615568639, 0.7646404115755685,
            0.04227222359126768, -0.2485828057615468, 0.8454821531441072, 0.7769855156051237,
            0.7342768488835001, 0.27889616300398723, 0.4680218977514422, 0.8246418191864404
        ]
        self.paper_pose = [
            -0.1220, 0.4, 0.6, -0.0769,
            0.0312, 0.4, 0.6, -0.0,
            0.1767, 0.4, 0.6, -0.0528,
            0.5284, 0.3693, 0.8977, 0.4863
        ]
        self.allegro_dof_lower = np.array([
            -0.4700, -0.1960, -0.1740, -0.2270,
             0.2630, -0.1050, -0.1890, -0.1620,
            -0.4700, -0.1960, -0.1740, -0.2270,
            -0.4700, -0.1960, -0.1740, -0.2270,
        ])
        self.allegro_dof_upper = np.array([
             0.4700, 1.6100, 1.7090, 1.6180,
             1.3960, 1.1630, 1.6440, 1.7190,
             0.4700, 1.6100, 1.7090, 1.6180,
             0.4700, 1.6100, 1.7090, 1.6180,
        ])

        # Data buffers (using deque for a sliding time window)
        self.window_maxlen = 300  # e.g., for 30 seconds at 10 Hz
        self.time_data = deque(maxlen=self.window_maxlen)
        self.command_data = [deque(maxlen=self.window_maxlen) for _ in range(16)]
        self.pose_data = [deque(maxlen=self.window_maxlen) for _ in range(16)]
        self.data_lock = threading.Lock()  # For thread-safe access
        self.running = True  # Flag to stop threads when needed

    def plot_loop(self):
        plt.ion()  # Turn on interactive plotting
        fig, axs = plt.subplots(16, 1, figsize=(10, 20), sharex=True)
        lines_command = []
        lines_pose = []
        for i in range(16):
            line_command, = axs[i].plot([], [], color='orange', label='Command')
            line_pose, = axs[i].plot([], [], color='blue', label='Current Pos')
            axs[i].set_ylabel(f'Joint {i}')
            axs[i].legend(loc='upper right')
            lines_command.append(line_command)
            lines_pose.append(line_pose)
        axs[-1].set_xlabel("Time (s)")

        while self.running:
            with self.data_lock:
                # Copy the data from deques for plotting
                tdata = list(self.time_data)
                cdata = [list(d) for d in self.command_data]
                pdata = [list(d) for d in self.pose_data]
            # Update each subplot's data
            for i in range(16):
                lines_command[i].set_data(tdata, cdata[i])
                lines_pose[i].set_data(tdata, pdata[i])
                axs[i].relim()
                axs[i].autoscale_view()
            plt.pause(0.05)
        plt.ioff()
        plt.show()

    def control_loop(self):
        # Note: Do not call rospy.init_node here.
        import rospy
        allegro = Allegro(hand_topic_prefix="allegroHand_0")
        rospy.sleep(0.5)

        hz = 10
        ros_rate = rospy.Rate(hz)

        # Setup: command the hand to the home pose
        for t in range(hz * 8):
            print("setup")
            allegro.command_joint_position(self.home_pose)
            obses, _ = allegro.poll_joint_position(wait=True)
            ros_rate.sleep()

        obses, _ = allegro.poll_joint_position(wait=True)
        print(f"Initial pose: {obses}")

        start_time = rospy.get_time()

        try:
            while not rospy.is_shutdown() and self.running:
                current_time = rospy.get_time()
                elapsed = current_time - start_time

                # Generate sinusoidal value between -1 and 1
                sin_val = np.sin(elapsed)
                # Start with the home_pose as baseline
                commands = np.array(self.home_pose)

                # --- TODO: Use the phase of the sinusoid instead of absolute time ---
                # Compute phase of the sine (period = 2π)
                phase = np.mod(elapsed, 2 * np.pi)
                # if phase < np.pi:
                # Pattern 1: update only joint index 3
                # idx = 5
                # lower = self.allegro_dof_lower[idx]
                # upper = self.allegro_dof_upper[idx]
                # commands[idx] = (sin_val * (upper - lower) + (upper + lower)) / 2
                # ---------------------------------------------------------------------
                # Pattern 2: update joints indices 3, 4, and 5
                for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
                    lower = self.allegro_dof_lower[idx]
                    upper = self.allegro_dof_upper[idx]
                    commands[idx] = (sin_val * (upper - lower) + (upper + lower)) / 2
                # ---------------------------------------------------------------------

                print(f"Commands: {commands}")
                allegro.command_joint_position(commands)
                ros_rate.sleep()

                obses, _ = allegro.poll_joint_position(wait=True)
                print(f"Current pose: {obses}")

                # Append new data (thread safe)
                with self.data_lock:
                    self.time_data.append(elapsed)
                    for i in range(16):
                        self.command_data[i].append(commands[i])
                        self.pose_data[i].append(obses[i])

        except KeyboardInterrupt:
            print("KeyboardInterrupt caught. Stopping hardware deployment.")
            self.running = False
            rospy.sleep(1)
            print("Hardware deployment stopped.")

    def deploy(self):
        import rospy
        # Initialize the ROS node in the main thread
        rospy.init_node("example", anonymous=True)
        # Start control and plotting threads
        plot_thread = threading.Thread(target=self.plot_loop)
        control_thread = threading.Thread(target=self.control_loop)

        plot_thread.start()
        control_thread.start()

        control_thread.join()
        # Signal the plot thread to exit if control loop ends
        self.running = False
        plot_thread.join()


if __name__ == "__main__":
    player = HardwarePlayerCalib()
    player.deploy()
