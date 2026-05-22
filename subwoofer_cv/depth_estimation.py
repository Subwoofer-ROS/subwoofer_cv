"""
Handles all code related to depth estimation, using the [Depth Anything](github.com/DepthAnything/Depth-Anything-V2) library.

This code is recommended to be offloaded onto a separate device for improved processing.
"""

# Message types
from sensor_msgs.msg import Image as Img
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import CameraInfo

# ROS libraries
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

# System libraries
import numpy as np
from transformers import pipeline
from PIL import Image

from sensor_msgs_py.point_cloud2 import create_cloud_xyz32


class DepthEstimator(Node):
    """
    Detects motion in a video stream.
    """

    def __init__(self):
        super().__init__("motion_detection")

        self.bridge = CvBridge()
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            trust_remote_code=True
        )

        self.image_sub = self.create_subscription(
            Img,
            "/image_raw",
            self.img_sub,
            10
        )

        self.caminfo_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self.caminfo_cb,
            10
        )

        self.depth_pub = self.create_publisher(
            Img,
            "/camera/depth/image_raw",
            10
        )

        self.pc_pub = self.create_publisher(
            PointCloud2,
            "/camera/points",
            10
        )

        self.cam_pub = self.create_publisher(
            CameraInfo,
            "/camera/depth/camera_info",
            10
        )

        self.last_camera_info = None
        self.have_cam_info = False
        self.fx = 1.0
        self.fy = 1.0
        self.cx = 0.0
        self.cy = 0.0

        self.get_logger().info(f"Depth estimator online!")

    def caminfo_cb(self, msg: CameraInfo):
        K = msg.k
        # K is row-major 3x3: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = K[0]
        self.fy = K[4]
        self.cx = K[2]
        self.cy = K[5]
        self.last_camera_info = msg
        self.have_cam_info = True

    def img_sub(self, msg: Image):
        array = self.bridge.imgmsg_to_cv2(msg)
        img = Image.fromarray(array)

        depth = self.pipe(img)["depth"]

        depth = np.array(depth)

        img = self.bridge.cv2_to_imgmsg(depth, encoding="8UC1")
        img.header = msg.header
        self.depth_pub.publish(img)
        if self.last_camera_info is not None:
            self.cam_pub.publish(self.last_camera_info)
        return

        depth = 255 - depth
        cv_img_32f = depth / 255.0

        h, w = cv_img_32f.shape[:2]

        if not self.have_cam_info:
            self.fx = float(max(w, h))
            self.fy = float(max(w, h))
            self.cx = w / 2.0
            self.cy = h / 2.0
        
        u = np.arange(w)
        v = np.arange(h)
        uu, vv = np.meshgrid(u, v)

        z = cv_img_32f
        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy

        points = np.dstack((x, y, z)).reshape(-1, 3)

        mask = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
        points = points[mask]
        cloud_msg = create_cloud_xyz32(
            msg.header,
            points.tolist()
        )
        self.pc_pub.publish(cloud_msg)
        


def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()