"""
keyboard_thrust_teleop.py

This script allows manual control of a robot's left and right thrusters via keyboard input.
It uses ROS 2 to publish Float64 messages to the specified thrust topics.

Controls:
W/S: Increase/Decrease forward thrust
A/D: Differential thrust for turning
SPACE: Stop all thrust
1: Set to max thrust
2: Set to zero thrust
Q or CTRL-C: Quit
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import sys
import termios
import tty
import signal

class KeyboardThrustTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_thrust_teleop')
        self.publisher_left = self.create_publisher(Float64, '/wamv/thrusters/left/thrust', 10)
        self.publisher_right = self.create_publisher(Float64, '/wamv/thrusters/right/thrust', 10)
        self.left_thrust = 0.0
        self.right_thrust = 0.0
        self.step = 50
        self.max_thrust = 200.0
        self.min_thrust = -200.0
        self.running = True
        self.print_instructions()

    def print_instructions(self):
        print("""
WAM-V Keyboard Thrust Control
-----------------------------
Reading from keyboard and Publishing to thrust topics

w/s : increase/decrease forward thrust for both thrusters
a/d : differential thrust (turn left/right)
space: stop all thrust
1    : increase to max thrust
2    : decrease to min thrust
q    : quit
CTRL-C to quit
-----------------------------
        """)

    def publish_thrust(self):
        msg_left = Float64()
        msg_right = Float64()
        msg_left.data = self.left_thrust
        msg_right.data = self.right_thrust
        self.publisher_left.publish(msg_left)
        self.publisher_right.publish(msg_right)
        print(f"\rLeft Thrust: {self.left_thrust:.2f}, Right Thrust: {self.right_thrust:.2f}  ", end='')

    def update_thrust(self, key):
        if key == 'w':
            self.left_thrust = min(self.left_thrust + self.step, self.max_thrust)
            self.right_thrust = min(self.right_thrust + self.step, self.max_thrust)
        elif key == 's':
            self.left_thrust = max(self.left_thrust - self.step, self.min_thrust)
            self.right_thrust = max(self.right_thrust - self.step, self.min_thrust)
        elif key == 'a':
            self.left_thrust = max(self.left_thrust - self.step, self.min_thrust)
            self.right_thrust = min(self.right_thrust + self.step, self.max_thrust)
        elif key == 'd':
            self.left_thrust = min(self.left_thrust + self.step, self.max_thrust)
            self.right_thrust = max(self.right_thrust - self.step, self.min_thrust)
        elif key == ' ':
            self.left_thrust = 0.0
            self.right_thrust = 0.0
        elif key == '1':
            self.left_thrust = self.max_thrust
            self.right_thrust = self.max_thrust
        elif key == '2':
            self.left_thrust = self.min_thrust
            self.right_thrust = self.min_thrust
        elif key == 'q':
            self.running = False
        else:
            print(f"\nUnknown key: {key}")

        self.publish_thrust()


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def signal_handler(sig, frame):
    print("\nExiting...")
    sys.exit(0)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardThrustTeleop()
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while node.running:
            key = get_key()
            node.update_thrust(key)
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
