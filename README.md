# ros-yolov5-node

For ROS2, uses ONNX GPU Runtime (Python SDK) to inference YOLOv5.

## Comparisons

- <https://github.com/ms-iot/ros_msft_onnx>: seems hardcoded to do post/pre-processing for specifically Tiny-YOLOv2 and caters to only object detection and object pose. Unfortunately isn't a library for running ONNX models in general then supplying your own pre/post-processing code.
- <https://github.com/raghavauppuluri13/yolov5_pytorch_ros>: requires Conda so is not `rosdep` compatible, plus PyTorch is huge. Does not support ROS2.
- <https://github.com/mats-robotics/yolov5_ros>: clones YOLOv5 from source as a submodule then needs to install its `requirements.txt` so not `rosdep` compatible. Does not support ROS2 either.

## Interesting

Rename to onnx_yolo_ros?

Why? Because implementation language & deep learning framework aside, the pre & post-processing code for YOLO models are largely the same between variants. Hence, once the model is exported/standardized to ONNX, it is possible to just swap which YOLO model you want to use. For example, from YOLOv5 to YOLOv4 or even YOLOv7.

That said, I have yet to thoroughly test the above. Even more so given YOLOv7 heavily uses YOLOv5's codebase. Separate implementations might be needed for different variant "generations".

## Note

Whenever you see size or shape, it has been converted to be `(width, height)`. Be mindful that numpy images are still `(height, width, channel)`, so you might want to apply `img.shape[1::-1]`.
