from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

PACKAGE_NAME = "onnx_yolov5_ros"

# By right, Nicepipe main launch file responsible for orchestration
# This demo file is just for convenience, hence the undeclared dependency
# aiortc_ros


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    namespace_arg = DeclareLaunchArgument(
        "namespace",
        description="Set the namespace of the nodes",
        default_value="/models",
    )

    yolov5_node = Node(
        package=PACKAGE_NAME,
        namespace=namespace,
        executable="yolov5",
        name="yolov5",
        # equivalent to --remap yolov5/frames_in:=/rtc/rtc_receiver/frames_out
        remappings=[
            ("~/frames_in", "/rtc/rtc_receiver/frames_out"),
            ("~/preds_out", "/data_out"),
        ],
    )

    aiortc_cfg = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory("aiortc_ros")) / "main.launch.py")
        ),
        launch_arguments=[("namespace", "/rtc")],
    )

    return LaunchDescription([namespace_arg, aiortc_cfg, yolov5_node])

