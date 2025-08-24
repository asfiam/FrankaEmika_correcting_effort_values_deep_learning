
#!/usr/bin/env python3

import sys
#import os

from isaacsim.robot.manipulators.examples.franka import Franka
##from omni.isaac.examples.base_sample import BaseSample
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np
##from omni.isaac.sensor.scripts.effort_sensor import EffortSensor
#import omni.graph.core as og

import rclpy
from isaacsim.examples.interactive.base_sample import BaseSample
from rclpy.node import Node
#from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import Header
from pxr import UsdPhysics
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from pxr import Gf
from pxr import UsdGeom

##from isaacsim.core.api.tasks import BaseTask
import omni.graph.core as og
import numpy as np
from omni.isaac.core.articulations import ArticulationView
import omni.usd
from scipy.spatial.transform import Rotation as R
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import omni.syntheticdata._syntheticdata as sd
import omni.replicator.core as rep
#from isaacsim.core.includes import getWorldTransformMatrix
#from omni.isaac.core.utils.numpy.rotations import matrix_to_quat

if not rclpy.ok():
    rclpy.init()

def gf_quat_to_np_array(quat):
    return np.array([quat.GetImaginary()[0], quat.GetImaginary()[1], quat.GetImaginary()[2], quat.GetReal()])



class FrankaEffortPublisher(Node):
    def __init__(self):
        super().__init__('franka_effort_publisher')
        self.effort_sensor_to_hand_publisher_ = self.create_publisher(WrenchStamped, '/franka_effort_sensor_to_hand', 10)
        self.effort_link7_to_sensor_publisher_ = self.create_publisher(WrenchStamped, '/franka_effort_link7_to_sensor', 10)
        
        self.EE_pose_publisher_ = self.create_publisher(PoseStamped, '/franka_EE_pose', 10)
                
class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self.ros_node = FrankaEffortPublisher()

        return

    def setup_scene(self):
        print("hello1")
        world = self.get_world()
        world.scene.add_default_ground_plane()
        #franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka", position=np.array([0.0, 0.0, 0.026])))
        #franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka", position=np.array([0.0, 0.0, 0.0])))
        franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka", position=np.array([0.0, 0.0, 0.01158])))
        #franka = world.scene.add(Franka(prim_path="/World/Fancy_Franka", name="fancy_franka", position=np.array([0.0, 0.0, 0.0162])))
        #cube = world.scene.add(DynamicCuboid(prim_path= "/World/Cube", name= "fancy_cube", position=np.array([0.0, 0.6, 0.0]), scale= np.array([0.4, 0.4, 0.4]) ))
        print("adding L object")
        L_obj_usd_path = "/home/digitaltwin/software/isaacsim/my_assets/L_object.usd"
        L_obj_prim_path = "/World/L_Object"
        add_reference_to_stage(L_obj_usd_path, L_obj_prim_path)
        print("added L object")
        
        #self.camera_floating = Camera(prim_path="/World/floating_camera",
        #    position=np.array([0.6267249122010365, -1.0899894150435772, 0.81929329203460317634038678793]),
        #    frequency=20, resolution=(256, 256),
        #    orientation=[-0.49814925, -0.06765672, -0.11633789, 0.85658356], #rot_utils.euler_angles_to_quats(np.array([58.06897, 0, -70306]),degrees=True),
        #)
        #self.camera_floating_prim = stage.GetPrimAtPath("/World/Fancy_Franka/floating_camera")
        self.camera_floating = Camera(prim_path="/World/Fancy_Franka/floating_camera",
                                        name='floating_camera')
        self.camera_wrist = Camera(prim_path="/World/Fancy_Franka/wrist_camera",
                                        name='wrist_camera')

        self.camera_floating.initialize()
        self.camera_wrist.initialize()
        print("hello 2")

        """
        in: pose
        transformation T 4x4 matrix
        apply T to pose
        T @ pose (T4x4, P4x1)
        pose -> object
        """

        self._create_action_graph_franka()
        #self._create_action_graph_obj()

        self.stage = omni.usd.get_context().get_stage()
        print("hello3")

        self.left_finger_prim = self.stage.GetPrimAtPath("/World/Fancy_Franka/panda_leftfinger")
        self.right_finger_prim = self.stage.GetPrimAtPath("/World/Fancy_Franka/panda_rightfinger")

        L_obj = self.stage.GetPrimAtPath("/World/L_Object")
        # L_obj_scale = L_obj.GetAttribute('xformOp:scale')
        # L_obj_scale.Set((100.0, 100.0, 100.0))

        # L_obj_xform = UsdGeom.Xformable(L_obj)
        # L_obj_xform.AddTranslateOp().Set((0.35, 0.1, 0.0))
        # L_obj_xform.AddScaleOp().Set((0.25, 0.25, 0.25))

        L_obj_xform = UsdGeom.Xformable(L_obj)
        L_obj_xform.AddTranslateOp().Set((0.39365, -0.07614, -0.00002))
        #L_obj_xform.AddTranslateOp().Set((0.39418, -0.07999, 0))
        L_obj_xform.AddScaleOp().Set((0.1, 0.1, 0.1))
        L_obj_xform.AddRotateXOp().Set(-0.11)   
        L_obj_xform.AddRotateYOp().Set(-0.011)  
        L_obj_xform.AddRotateZOp().Set(90.0)   

        print("hello4")

        L_obj_mass_api = UsdPhysics.MassAPI.Apply(L_obj)
        L_obj_mass_api.CreateMassAttr(0.341)
        print("hello5")
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)

        # for _ in range(10):
        #     await self._world.step_async()

        self.publish_rgb(self.camera_floating, freq=20)
        self.publish_depth(self.camera_floating, freq=20)

        self.publish_rgb(self.camera_wrist, freq=20)
        self.publish_depth(self.camera_wrist, freq=20)

        #await self._world.play_async()
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        #await self._world.play_async()
        return

    def physics_step(self, step_size):
        print("inside physics step")
        self.get_observations()

        # self.publish_rgb(self.camera_floating, freq=20)
        # self.publish_depth(self.camera_floating, freq=20)

        # self.publish_rgb(self.camera_wrist, freq=20)
        # self.publish_depth(self.camera_wrist, freq=20)
        return

    def get_observations(self):
        print("i am inside observations")
        world = self.get_world()
        franka = world.scene.get_object("fancy_franka")
        articulation_view = franka._articulation_view
        sensor_joint_forces = franka.get_measured_joint_forces()
        for i, f in enumerate(sensor_joint_forces):
            formatted = [f"{float(x):.6f}" for x in f]   # format each element
            print(f"Joint {i}: {formatted}")
        #print(f"Sensor joint forces: {sensor_joint_forces}")
        joint_link_id = {}

        for prim in self.stage.Traverse():
            if prim.GetTypeName() == "PhysicsFixedJoint" or prim.GetTypeName() == "PhysicsRevoluteJoint" or prim.GetTypeName() == "PhysicsPrismaticJoint":
                joint = UsdPhysics.Joint(prim)
                joint_name = prim.GetName()
                targets = joint.GetBody1Rel().GetTargets()
                if not targets:
                    continue
                link_path = targets[0]
                link_prim = self.stage.GetPrimAtPath(link_path)
                if not link_prim.IsValid():
                    continue
                link_name = link_prim.GetName()
                try:
                    link_index = articulation_view.get_link_index(link_name)
                    joint_link_id[joint_name] = link_index
                except Exception as e:
                    print(f"Could not resolve link index for joint '{joint_name}': {e}") 
        print(joint_link_id)

        joint_names = franka._articulation_view.joint_names

        # Find the array indices for the two joints you want
        idx_sensor_joint = joint_names.index("link7_to_sensor")  # between link 7 & sensor
        idx_fixed_joint  = joint_names.index("sensor_to_hand")  

        print(f"index for joint link7 and sensor {idx_sensor_joint}")
        print(f"index for sensor and hand {idx_fixed_joint}")

        ####################################### publishing EFFORT / FORCE #######################################

        # Publishing force and torque at joint 4 (example: end effector)
        if len(sensor_joint_forces) >= 4:
            #print("inside if statement 1")
            force_torque_sensor_to_hand = sensor_joint_forces[8]
            force_torque_link7_to_sensor = sensor_joint_forces[7]
            print(f"sensor to hand {force_torque_sensor_to_hand}")
            print(f"link7 to sensor {force_torque_link7_to_sensor}")
            # force_torque_sensor_to_hand = sensor_joint_forces[9]
            # force_torque_link7_to_sensor = sensor_joint_forces[8]

            #print(f"force torque = {force_torque}")
            #msg = self.ros_node.wrench_stamped_msg

            msg_sensor_to_hand = WrenchStamped()
            msg_link7_to_sensor = WrenchStamped()

            msg_sensor_to_hand.header = Header()
            msg_sensor_to_hand.header.stamp = self.ros_node.get_clock().now().to_msg()
            #print("inside if statement 2")
            msg_sensor_to_hand.wrench.force.x = float(force_torque_sensor_to_hand[0])
            msg_sensor_to_hand.wrench.force.y = float(force_torque_sensor_to_hand[1])
            msg_sensor_to_hand.wrench.force.z = float(force_torque_sensor_to_hand[2])
            msg_sensor_to_hand.wrench.torque.x = float(force_torque_sensor_to_hand[3])
            msg_sensor_to_hand.wrench.torque.y = float(force_torque_sensor_to_hand[4])
            msg_sensor_to_hand.wrench.torque.z = float(force_torque_sensor_to_hand[5])
            #print(f"Message is: {msg}")
            #print(f"Message: {msg.wrench}")
            #print("inside if statement 3")

            msg_link7_to_sensor.header = Header()
            msg_link7_to_sensor.header.stamp = self.ros_node.get_clock().now().to_msg()
            #print("inside if statement 2")
            msg_link7_to_sensor.wrench.force.x = float(force_torque_link7_to_sensor[0])
            msg_link7_to_sensor.wrench.force.y = float(force_torque_link7_to_sensor[1])
            msg_link7_to_sensor.wrench.force.z = float(force_torque_link7_to_sensor[2])
            msg_link7_to_sensor.wrench.torque.x = float(force_torque_link7_to_sensor[3])
            msg_link7_to_sensor.wrench.torque.y = float(force_torque_link7_to_sensor[4])
            msg_link7_to_sensor.wrench.torque.z = float(force_torque_link7_to_sensor[5])


            '''
            msg.wrench.force.x = force_torque[0]
            msg.wrench.force.y = force_torque[1]
            msg.wrench.force.z = force_torque[2]
            msg.wrench.torque.x = force_torque[3]
            msg.wrench.torque.y = force_torque[4]
            msg.wrench.torque.z = force_torque[5]
            '''
            self.ros_node.effort_sensor_to_hand_publisher_.publish(msg_sensor_to_hand)
            self.ros_node.effort_link7_to_sensor_publisher_.publish(msg_link7_to_sensor)

        ####################################### publishing EE POSE #######################################

        #print(left_finger_prim)
        #print(right_finger_prim.GetPropertyNames())
        

        #left_finger_prim = stage.GetPrimAtPath(self.left_finger_path)
        #tf_left_usd = omni.usd.get_world_transform_matrix(left_finger_prim)
        #print(tf_left_usd)

        #right_finger_prim = stage.GetPrimAtPath(self.right_finger_path)
        #tf_right_usd = omni.usd.get_world_transform_matrix(right_finger_prim)
        #print(tf_right_usd)

        right_finger_translate_op = self.right_finger_prim.GetAttribute('xformOp:translate')
        left_finger_translate_op = self.left_finger_prim.GetAttribute('xformOp:translate')
        EE_orient_op = self.left_finger_prim.GetAttribute('xformOp:orient')
        EE_orient = gf_quat_to_np_array(EE_orient_op.Get())
        EE_translate = (right_finger_translate_op.Get() + left_finger_translate_op.Get())/2

        #print(f"orient: {EE_orient}")
        #print(f"translate: {EE_translate}")

        #tf_left = np.array(tf_left_usd)
        #tf_right = np.array(tf_right_usd)

        #pos_left = tf_left[:3, 3]
        #pos_right = tf_right[:3, 3]
        #pos_avg = (pos_left + pos_right) / 2.0
        #print(f"pos avg {pos_avg}")

        #rot_left = tf_left[:3, :3]
        #rot_right = tf_right[:3, :3]
        #rot_avg = (rot_left + rot_right) / 2.0  # crude average
        #quat_avg = omni.usd.matrix_to_quat(rot_avg)  # [x, y, z, w]
        #quat_avg = convert_quat(rot_avg, to='xyzw')
        #quat_avg = R.from_matrix(rot_avg).as_quat()  # returns [x, y, z, w]

        # Fill PoseStamped message
        msg_EE_pose = PoseStamped()
        msg_EE_pose.header.stamp = self.ros_node.get_clock().now().to_msg()
        msg_EE_pose.header.frame_id = "world"
        msg_EE_pose.pose.position.x = float(EE_translate[0])
        msg_EE_pose.pose.position.y = float(EE_translate[1])
        msg_EE_pose.pose.position.z = float(EE_translate[2])
        msg_EE_pose.pose.orientation.x = float(EE_orient[0])
        msg_EE_pose.pose.orientation.y = float(EE_orient[1])
        msg_EE_pose.pose.orientation.z = float(EE_orient[2])
        msg_EE_pose.pose.orientation.w = float(EE_orient[3])

        self.ros_node.EE_pose_publisher_.publish(msg_EE_pose)     
        print("Published effort for joint 4 (end effector).")
        return

    def _create_action_graph_franka(self):
        #import omni.graph.core as og

        og.Controller.edit(
            {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                    ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                    ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),

                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),

                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                    ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    # Providing path to /panda robot to Articulation Controller node
                    # Providing the robot path is equivalent to setting the targetPrim in Articulation Controller node
                    # ("ArticulationController.inputs:usePath", True),      # if you are using an older version of Isaac Sim, you may need to uncomment this line
                    ("ArticulationController.inputs:robotPath", "/World/Fancy_Franka"),
                    ("PublishJointState.inputs:targetPrim", "/World/Fancy_Franka"),
                    #("SubscribeJointState.inputs:topicName", "/panda_teleop/joint_states_real"),
                    ("SubscribeJointState.inputs:topicName", "/joint_command"),
                ],
            },
        )

    def _create_action_graph_obj(self):
        og.Controller.edit(
            {"graph_path": "/ActionGraph_Object", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("ROS2Subscriber", "isaacsim.ros2.bridge.ROS2Subscriber"),
                    ("MakeVector3", "omni.graph.nodes.MakeVector3"),
                    ("MakeVector4", "omni.graph.nodes.MakeVector4"),
                    ("WritePrimAttribute1", "omni.graph.nodes.WritePrimAttribute"),
                    ("WritePrimAttribute2", "omni.graph.nodes.WritePrimAttribute"),
                ],

                og.Controller.Keys.SET_VALUES: [
                    ("ROS2Subscriber.inputs:messagePackage", "geometry_msgs"),
                    ("ROS2Subscriber.inputs:messageName", "PoseStamped"),
                    ("ROS2Subscriber.inputs:topicName", "/icg_tracker_2"),

                    ("WritePrimAttribute1.inputs:prim", "/World/L_Object"),
                    ("WritePrimAttribute1.inputs:name", "xformOp:translate"),

                    ("WritePrimAttribute2.inputs:prim", "/World/L_Object"),
                    ("WritePrimAttribute2.inputs:name", "xformOp:orient"),
                ],

                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "ROS2Subscriber.inputs:execIn"),
                    ("ROS2Context.outputs:context", "ROS2Subscriber.inputs:context"),
                    ("ROS2Subscriber.outputs:execOut", "WritePrimAttribute1.inputs:execIn"),
                    ("ROS2Subscriber.outputs:execOut", "WritePrimAttribute2.inputs:execIn"),
                    ("MakeVector3.outputs:tuple", "WritePrimAttribute1.inputs:value"),
                    ("MakeVector4.outputs:tuple", "WritePrimAttribute2.inputs:value"),
                    #("ROS2Subscriber.outputs:pose:position:x", "MakeVector3.inputs:x"),
                ],

            },
        )

        print("nodes created and values set")

    def world_cleanup(self):
        #print("5. inside clean world")
        return

    ####################################### publishing RGB values from both cameras #######################################

    def publish_rgb(self, camera: Camera, freq):
        print("i am inside publish_rgb")
        # The following code will link the camera's render product and publish the data to the specified topic name.
        render_product = camera._render_product_path
        step_size = int(60/freq)
        topic_name = camera.name+"_rgb"
        queue_size = 1
        node_namespace = ""
        frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

        rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
        writer = rep.writers.get(rv + "ROS2PublishImage")
        writer.initialize(
            frameId=frame_id,
            nodeNamespace=node_namespace,
            queueSize=queue_size,
            topicName=topic_name
        )
        writer.attach([render_product])

        # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
        gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv + "IsaacSimulationGate", render_product
        )
        og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

        return
    

    ####################################### publishing DEPTH values from both cameras #######################################

    def publish_depth(self, camera: Camera, freq):
        print("i am inside publish_depth")
        # The following code will link the camera's render product and publish the data to the specified topic name.
        render_product = camera._render_product_path
        step_size = int(60/freq)
        topic_name = camera.name+"_depth"
        queue_size = 1
        node_namespace = ""
        frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

        rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                                sd.SensorType.DistanceToImagePlane.name
                            )
        writer = rep.writers.get(rv + "ROS2PublishImage")
        writer.initialize(
            frameId=frame_id,
            nodeNamespace=node_namespace,
            queueSize=queue_size,
            topicName=topic_name
        )
        writer.attach([render_product])

        # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
        gate_path = omni.syntheticdata.SyntheticData._get_node_path(
            rv + "IsaacSimulationGate", render_product
        )
        og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

        return
     
'''def main(args=None):
    print(" inside MAIN")
    rclpy.init(args=args)
    rclpy.spin(FrankaEffortPublisher())
    rclpy.shutdown()
'''





'''
# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim.examples.interactive.base_sample import BaseSample
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Header
# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):

        world = self.get_world()
        world.scene.add_default_ground_plane()
        return

    async def setup_post_load(self):
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
'''


"""

                    ("WritePrimAttribute1", "omni.replicator.core.OgnWritePrimAttribute"),
                    ("WritePrimAttribute2", "omni.replicator.core.OgnWritePrimAttribute"),
"""

"""                 og.Controller.Keys.SET_VALUES: [
                    ("ROS2Subscriber.inputs:messagePackage", "geometry_msgs"),
                    ("ROS2Subscriber.inputs:messageName", "PoseStamped"),
                    ("ROS2Subscriber.inputs:topicName", "/icg_tracker_2"),

                    #("WritePrimAttribute1.inputs:prims", "/World/Cube"),
                    #("WritePrimAttribute1.inputs:attributeType", "xformOp:translate"),

                    #("WritePrimAttribute2.inputs:prims", "/World/Cube"),
                    #("WritePrimAttribute2.inputs:attributeType", "xformOp:orient"),
                ], """

"""
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "ROS2Subscriber.inputs:execIn"),
                    ("ROS2Context.outputs:context", "ROS2Subscriber.inputs:context"),

                    ("ROS2Subscriber.outputs:execOut", "WritePrimAttribute1.inputs:execIn"),
                    ("ROS2Subscriber.outputs:execOut", "WritePrimAttribute2.inputs:execIn"),
                    ("ROS2Subscriber.outputs:orientation:w", "MakeVector4.inputs:w"),
                    ("ROS2Subscriber.outputs:orientation:x", "MakeVector4.inputs:x"),
                    ("ROS2Subscriber.outputs:orientation:y", "MakeVector4.inputs:y"),
                    ("ROS2Subscriber.outputs:orientation:z", "MakeVector4.inputs:z"),
                    ("ROS2Subscriber.outputs:position:x", "MakeVector3.inputs:x"),
                    ("ROS2Subscriber.outputs:position:y", "MakeVector3.inputs:y"),
                    ("ROS2Subscriber.outputs:position:z", "MakeVector3.inputs:z"),

                    ("MakeVector3.outputs:tuple", "WritePrimAttribute1.inputs:values"),
                    ("MakeVector4.outputs:tuple", "WritePrimAttribute2.inputs:values"),
                ],"""