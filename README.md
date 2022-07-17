# ros-yolov5-node

For ROS2, uses ONNX GPU Runtime (Python SDK) to inference YOLOv5.

## Comparisons

- <https://github.com/ms-iot/ros_msft_onnx>: seems hardcoded to do post/pre-processing for specifically Tiny-YOLOv2 and caters to only object detection and object pose. Unfortunately isn't a library for running ONNX models in general then supplying your own pre/post-processing code.
- <https://github.com/raghavauppuluri13/yolov5_pytorch_ros>: requires Conda so is not `rosdep` compatible, plus PyTorch is huge. Does not support ROS2.
- <https://github.com/mats-robotics/yolov5_ros>: clones YOLOv5 from source as a submodule then needs to install its `requirements.txt` so not `rosdep` compatible. Does not support ROS2 either.

## Interesting

TODO: rename to onnx_yolo_ros

Why? Because implementation language & deep learning framework aside, the pre & post-processing code for YOLO models are largely the same between variants. Hence, once the model is exported/standardized to ONNX, it is possible to just swap which YOLO model you want to use. For example, from YOLOv5 to YOLOv4 or even YOLOv7.

## TODO

Should we write a separate python onnxruntime library for nodes to inherit from? Does that go into ros-nice-node? Or is it not too extra because turns out sticking pre & post processing on a model is easy? Should we NOT use ros-nice-node? Maybe its a bad idea if you intend this packages to actually be used or standalone because ros-nice-node is too ambigious a dependency to ever be added to `rosdep` and I bet they hate it if you were to create a utilities package like this instead of being standalone...
