import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading
import time

import rospy
from std_msgs.msg import String, Float32
from sensor_msgs.msg import JointState

# Initial selected motor (this will be updated during control loop)
SELECTED_MOTOR = [0]
RATIO = 0.5  # This is not used anymore since we use unscaled command values.
motor_name = {0:"index_0",
              1:"index_1",
              2:"index_2",
              3:"index_3",
              4:"middle_0",
              5:"middle_1",
              6:"middle_2",
              7:"middle_3",
              8:"ring_0",
              9:"ring_1",
              10:"ring_2",
              11:"ring_3",
              12:"thumb_0",
              13:"thumb_1",
              14:"thumb_2",
              15:"thumb_3",
             }

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
        # Initial hand poses (joint positions)
        # self.home_pose = [
        #     0.018135, -0.257183, 0.729801, 0.761141,
        #     0.049130, -0.250311, 0.739642, 0.764640,
        #     0.042272, -0.248582, 0.845482, 0.776985,
        #     0.734276, 0.2788961, 0.468021, 0.824641,
        # ]
        self.allegro_dof_lower = np.array([
            -0.4700, -0.1960, -0.1740, -0.2270,
            -0.4700, -0.1960, -0.1740, -0.2270,
            -0.4700, -0.1960, -0.1740, -0.2270,
             0.2630, -0.1050, -0.1890, -0.1620,
        ])
        self.allegro_dof_upper = np.array([
             0.4700, 1.6100, 1.7090, 1.6180,
             0.4700, 1.6100, 1.7090, 1.6180,
             0.4700, 1.6100, 1.7090, 1.6180,
             1.3960, 1.1630, 1.6440, 1.7190,
        ])

        # Data buffers (using deque for a sliding time window)
        self.window_maxlen = 800  # e.g., 30 seconds at 10 Hz
        self.time_data = deque(maxlen=self.window_maxlen)
        self.command_data = [deque(maxlen=self.window_maxlen) for _ in range(16)]
        self.pose_data = [deque(maxlen=self.window_maxlen) for _ in range(16)]
        self.data_lock = threading.Lock()  # For thread-safe access
        self.running = True  # Flag to stop threads

        # Record maximum error for each motor
        self.max_error = np.zeros(16)

    def plotting_loop(self):
        """
        Modified plotting loop to plot data only for the current motor.
        It creates a single figure with two subplots:
         - The top subplot shows commanded (orange) vs measured (blue) positions.
         - The bottom subplot shows the positional error (red).
        """
        plt.ion()  # Turn on interactive mode

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.set_title("Current Motor: Commanded (orange) vs Measured (blue)")
        ax2.set_title("Current Motor: Position Error (red)")
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Initialize the lines.
        line_command, = ax1.plot([], [], color='orange', label='Command')
        line_pose, = ax1.plot([], [], color='blue', label='Measured')
        ax1.legend(loc='upper right')
        line_error, = ax2.plot([], [], color='red', label='Error')
        ax2.legend(loc='upper right')
        ax2.set_xlabel("Time (s)")

        while self.running:
            with self.data_lock:
                tdata = list(self.time_data)
            # Read the current motor from the global SELECTED_MOTOR variable.
            current_motor = SELECTED_MOTOR[0]
            # Extract data only for the current motor.
            cdata = list(self.command_data[current_motor])
            pdata = list(self.pose_data[current_motor])
            err = np.abs(np.array(cdata) - np.array(pdata))
            
            # Update the plot data.
            line_command.set_data(tdata, cdata)
            line_pose.set_data(tdata, pdata)
            line_error.set_data(tdata, err)

            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()

            plt.pause(0.05)

        plt.ioff()
        plt.close('all')

    def control_loop(self):
        # Do not call rospy.init_node here.
        allegro = Allegro(hand_topic_prefix="allegroHand_0")
        rospy.sleep(0.5)

        hz = 10  # 10 Hz control loop
        ros_rate = rospy.Rate(hz)

        start_time = rospy.get_time()

        for t in range(hz * 8):
            print("Setting up home pose...")
            allegro.command_joint_position(self.allegro_dof_lower)
            obses, _ = allegro.poll_joint_position(wait=True)
            elapsed = rospy.get_time() - start_time
            # Record data for all motors.
            with self.data_lock:
                self.time_data.append(elapsed)
                for i in range(16):
                    self.command_data[i].append(self.allegro_dof_lower[i])
                    self.pose_data[i].append(obses[i])
            ros_rate.sleep()

        # No torque mode: hold current positions for 4 seconds.
        for t in range(hz * 8):
            print("No torque mode iteration: holding current positions...")
            obses, _ = allegro.poll_joint_position(wait=True)
            allegro.command_joint_position(obses)
            elapsed = rospy.get_time() - start_time
            # Record data (command equals measured positions).
            with self.data_lock:
                self.time_data.append(elapsed)
                for i in range(16):
                    self.command_data[i].append(obses[i])
                    self.pose_data[i].append(obses[i])
            ros_rate.sleep()

        obses, _ = allegro.poll_joint_position(wait=True)
        print(f"Initial pose: {obses}")

        # Clear the data buffers so that only the response to the step command is recorded.
        with self.data_lock:
            self.time_data.clear()
            for i in range(16):
                self.command_data[i].clear()
                self.pose_data[i].clear()

        loop_count = 0  # Count each control loop (10 Hz)

        try:
            while not rospy.is_shutdown() and self.running:
                # Use a single upper step command
                unscaled_command = 0.8

                global SELECTED_MOTOR
                current_motor = SELECTED_MOTOR[0]
                L = self.allegro_dof_lower[current_motor]
                U = self.allegro_dof_upper[current_motor]
                command_value = (U - L) / 2.0 * unscaled_command + (U + L) / 2.0

                print(f"Loop: {loop_count}, Motor: {motor_name[current_motor]}, "
                    f"Command: {unscaled_command} (Scaled to {command_value:.4f})")

                # For non-selected motors, command them to hold their current positions.
                current_positions, _ = allegro.poll_joint_position(wait=True)
                if current_positions is None:
                    current_positions = np.array(self.home_pose)
                commands = np.copy(current_positions)
                # Update only the current_motor with the desired command_value.
                commands[current_motor] = command_value

                allegro.command_joint_position(commands)
                ros_rate.sleep()
                obses, _ = allegro.poll_joint_position(wait=True)

                # Record maximum error for each motor.
                errors = np.abs(np.array(commands) - np.array(obses))
                for i in range(16):
                    self.max_error[i] = max(self.max_error[i], errors[i])

                elapsed = rospy.get_time() - start_time
                with self.data_lock:
                    self.time_data.append(elapsed)
                    for i in range(16):
                        self.command_data[i].append(commands[i])
                        self.pose_data[i].append(obses[i])
                loop_count += 1

                print(f"Max error (degrees): {np.round(self.max_error * 180/np.pi, 1)}")

                # TODO: Using self.pose_data record, find the period time peak-to-peak.
                # We use the recorded time and measured position for the current motor.
                with self.data_lock:
                    t_arr = np.array(self.time_data)
                    p_arr = np.array(self.pose_data[current_motor])
                if len(p_arr) > 3:
                    first_peak_idx = None
                    # Find the first local maximum.
                    for j in range(1, len(p_arr) - 1):
                        if p_arr[j] > p_arr[j - 1] and p_arr[j] > p_arr[j + 1]:
                            first_peak_idx = j
                            break
                    if first_peak_idx is not None:
                        # Calculate time to the first peak relative to the start of the recording.
                        time_to_first_peak = t_arr[first_peak_idx] - t_arr[0]
                        print("Time to first peak for motor {}: {:.4f} sec".format(
                            motor_name[current_motor], time_to_first_peak))


        except KeyboardInterrupt:
            print("KeyboardInterrupt caught in control_loop. Stopping hardware deployment.")
            self.running = False
            rospy.sleep(1)
        finally:
            self.running = False
            # Print the recorded maximum errors for each motor.
            print("Maximum errors recorded for each motor:")
            for i in range(16):
                print(f"Motor {motor_name[i]}: {self.max_error[i]:.4f}")



    def deploy(self):
        rospy.init_node("example", anonymous=True)
        plotting_thread = threading.Thread(target=self.plotting_loop)
        control_thread = threading.Thread(target=self.control_loop)

        plotting_thread.start()
        control_thread.start()

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught in deploy. Stopping all threads...")
        finally:
            self.running = False
            rospy.signal_shutdown("KeyboardInterrupt")
            plt.close('all')
            control_thread.join()
            plotting_thread.join()


if __name__ == "__main__":
    player = HardwarePlayerCalib()
    player.deploy()
