from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

PACKAGE_NAME = "onnx_yolov5_ros"

# By right, Nicepipe main launch file responsible for orchestration
# This demo file is just for convenience, hence the undeclared dependency
# aiortc_ros


def generate_launch_description():
    yolov5_node = Node(
        package=PACKAGE_NAME,
        namespace="/models",
        executable="yolov5",
        name="yolov5",
        # equivalent to --remap yolov5/frames_in:=/rtc/rtc_receiver/frames_out
        remappings=[
            ("~/frames_in", "/rtc/rtc_receiver/frames_out"),
            ("~/preds_out", "/data_out"),
        ],
        respawn=True,
    )

    aiortc_cfg = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory("aiortc_ros")) / "main.launch.py")
        ),
        launch_arguments=[("namespace", "/rtc")],
    )

    return LaunchDescription([yolov5_node, aiortc_cfg])

