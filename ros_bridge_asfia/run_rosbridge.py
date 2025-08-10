#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import roslibpy
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import WrenchStamped



class Ros1ToRos2PoseBridge(Node):
    def __init__(self,
                 ros1_host='192.168.3.98',
                 ros1_port=9090,
                 ros1_pose_topic='/icg_tracker_1/pose',
                 ros2_pose_topic='/icg_tracker_2',
                 ros1_wrench_topic='/bus0/ft_sensor0/ft_sensor_readings/wrench',
                 ros2_wrench_topic='/franka_effort_real'):
        super().__init__('ros1_to_ros2_pose_bridge')

        # ROS 2 publishers
        #self.pose_publisher_ = self.create_publisher(PoseStamped, ros2_pose_topic, 10)
        self.wrench_publisher_ = self.create_publisher(WrenchStamped, ros2_wrench_topic, 10)

        #self.get_logger().info(f'Publishing Pose to ROS 2 POSE topic: {ros2_pose_topic}')
        self.get_logger().info(f'Publishing Wrench to ROS 2 WRENCH topic: {ros2_wrench_topic}')

        # Connect to ROS 1
        self.ros1 = roslibpy.Ros(host=ros1_host, port=ros1_port)
        self.ros1.on_ready(lambda: self.get_logger().info(f'Connected to ROS 1 bridge at {ros1_host}:{ros1_port}'))
        self.ros1.run()

        # ROS 1 subscribers
        #self.ros1_pose_listener = roslibpy.Topic(self.ros1, ros1_pose_topic, 'geometry_msgs/PoseStamped')
        #self.ros1_pose_listener.subscribe(self.ros1_pose_callback)

        self.ros1_wrench_listener = roslibpy.Topic(self.ros1, ros1_wrench_topic, 'geometry_msgs/WrenchStamped')
        self.ros1_wrench_listener.subscribe(self.ros1_wrench_callback)

        #self.get_logger().info(f'Subscribed to ROS 1 POSE topic: {ros1_pose_topic}')
        self.get_logger().info(f'Subscribed to ROS 1 WRENCH topic: {ros1_wrench_topic}')

        # Define static camera-to-robot transform (calibration)
        self.T_cam_to_robot = np.linalg.inv(
            self.build_transform(
            pos=[0.525557, -0.357399, 0.263263],
            quat=[-0.195281, 0.227952, 0.667022, 0.681898] # x, y, z, w
        )
        )

    def build_transform(self, pos, quat):
        """Create a 4x4 transformation matrix from position and quaternion."""
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat, scalar_first=False).as_matrix()
        T[:3, 3] = pos
        return T

    def ros1_pose_callback(self, msg):
        """Callback to receive ROS 1 pose and publish transformed pose to ROS 2."""

        # Extract pose from ROS 1 message
        pos = np.array([
            msg['pose']['position']['x'],
            msg['pose']['position']['y'],
            msg['pose']['position']['z']
        ])
        quat = np.array([
            msg['pose']['orientation']['x'],
            msg['pose']['orientation']['y'],
            msg['pose']['orientation']['z'],
            msg['pose']['orientation']['w']
        ])

        T_pose = self.build_transform(pos, quat)

        R_correction = R.from_euler('z', -90, degrees=True)
        T_correction = self.build_transform(pos=[0, 0, -0.522], quat=R_correction.as_quat(scalar_first=False))

        # Transform to robot frame
        T_transformed = T_correction @ self.T_cam_to_robot @ T_pose

        transformed_pos = T_transformed[:3, 3]
        transformed_quat = R.from_matrix(T_transformed[:3, :3]).as_quat()  # x, y, z, w

        # Build ROS 2 PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'panda_link0'  # frame of reference
        pose_msg.header.stamp.sec = msg['header']['stamp']['secs']
        pose_msg.header.stamp.nanosec = msg['header']['stamp']['nsecs']

        pose_msg.pose.position.x = float(transformed_pos[0])
        pose_msg.pose.position.y = float(transformed_pos[1])
        pose_msg.pose.position.z = float(transformed_pos[2])

        pose_msg.pose.orientation.x = float(transformed_quat[0])
        pose_msg.pose.orientation.y = float(transformed_quat[1])
        pose_msg.pose.orientation.z = float(transformed_quat[2])
        pose_msg.pose.orientation.w = float(transformed_quat[3])

        self.publisher_.publish(pose_msg)
        self.get_logger().debug('Published transformed PoseStamped to ROS 2.')

    def ros1_wrench_callback(self, msg):
        ros2_msg = WrenchStamped()

        ros2_msg.header.stamp = self.get_clock().now().to_msg()
        ros2_msg.header.frame_id = msg['header']['frame_id']

        # Fill wrench force
        ros2_msg.wrench.force.x = msg['wrench']['force']['x']
        ros2_msg.wrench.force.y = msg['wrench']['force']['y']
        ros2_msg.wrench.force.z = msg['wrench']['force']['z']

        # Fill wrench torque
        ros2_msg.wrench.torque.x = msg['wrench']['torque']['x']
        ros2_msg.wrench.torque.y = msg['wrench']['torque']['y']
        ros2_msg.wrench.torque.z = msg['wrench']['torque']['z']

        self.wrench_publisher_.publish(ros2_msg)

    
    def destroy_node(self):
        self.get_logger().info('Shutting down bridge...')
        self.ros1_listener.unsubscribe()
        self.ros1.terminate()
        super().destroy_node()


def main():
    rclpy.init()
    node = Ros1ToRos2PoseBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt. Exiting...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()




    # def ros1_callback(self, message):
    #     pose_stamped = PoseStamped()
    #     #pose0 = Pose()
    #     #print("I am here 2")

    #     # Convert header
        
    #     pose_stamped.header.frame_id = message['header']['frame_id']
    #     pose_stamped.header.stamp.sec = message['header']['stamp']['secs']
    #     pose_stamped.header.stamp.nanosec = message['header']['stamp']['nsecs']

    #     # Convert pose
    #     pose_stamped.pose.position.x = message['pose']['position']['x']
    #     pose_stamped.pose.position.y = message['pose']['position']['y']
    #     pose_stamped.pose.position.z = message['pose']['position']['z']

    #     pose_stamped.pose.orientation.x = message['pose']['orientation']['x']
    #     pose_stamped.pose.orientation.y = message['pose']['orientation']['y']
    #     pose_stamped.pose.orientation.z = message['pose']['orientation']['z']
    #     pose_stamped.pose.orientation.w = message['pose']['orientation']['w'] 

    #     self.publisher_.publish(pose_stamped)
    #     #print("I am here 3")
    #     self.get_logger().debug('Published message to ROS 2.')