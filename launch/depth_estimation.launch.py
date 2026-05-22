# Launch file for depth estimation

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    depth_estimator = Node(
        package="subwoofer_cv",
        executable="DepthEstimation"
    )
    ld.add_action(depth_estimator)

    depth_to_pc = Node(
        package="depth_image_proc",
        executable="point_cloud_xyz_node",
        remappings=[
            ("image_rect", "/camera/depth/image_raw"),
            ("camera_info", "/camera/depth/camera_info"),
            ("points", "/camera/depth/points"),
        ]
    )
    ld.add_action(depth_to_pc)

    return ld