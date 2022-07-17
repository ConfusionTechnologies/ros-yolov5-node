from __future__ import annotations
from dataclasses import dataclass, field
import sys
from copy import copy
from ast import literal_eval

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from cv_bridge import CvBridge

import numpy as np
from onnxruntime import InferenceSession

from sensor_msgs.msg import Image, CompressedImage
from nicefaces.msg import BBox2D, Pred2D, Pred2DArray
from foxglove_msgs.msg import ImageMarkerArray
from visualization_msgs.msg import ImageMarker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from nicepynode import Job, JobCfg
from onnx_yolov5_ros.processing import letterbox, non_max_suppression, scale_coords

NODE_NAME = "yolov5_model"

cv_bridge = CvBridge()

# Realtime Profile: don't bog down publisher when model is slow
rt_profile = copy(QoSPresetProfiles.SENSOR_DATA.value)
rt_profile.depth = 0


@dataclass
class YoloV5Cfg(JobCfg):
    model_path: str = "/code/models/yolov7-w6.onnx"

    frames_in_topic: str = "~/frames_in"
    preds_out_topic: str = "~/preds_out"
    markers_topic: str = "~/bbox_markers"

    onnx_providers: list[str] = field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    img_hw: tuple[int, int] = (640, 640)
    accept_compression: bool = False
    """Only necessary for 4K. Before that, performance hit from compression > bandwidth hit."""


@dataclass
class YoloV5Predictor(Job[YoloV5Cfg]):

    ini_cfg: YoloV5Cfg = field(default_factory=YoloV5Cfg)

    def attach_params(self, node: Node, cfg: YoloV5Cfg):
        super(YoloV5Predictor, self).attach_params(node, cfg)

        node.declare_parameter("frames_in_topic", cfg.frames_in_topic)
        node.declare_parameter("preds_out_topic", cfg.preds_out_topic)

    def attach_behaviour(self, node: Node, cfg: YoloV5Cfg):
        super(YoloV5Predictor, self).attach_behaviour(node, cfg)

        self._frames_sub = node.create_subscription(
            CompressedImage if cfg.accept_compression else Image,
            cfg.frames_in_topic,
            self._on_input,
            rt_profile,
        )
        self._pred_pub = node.create_publisher(Pred2DArray, cfg.preds_out_topic, 5)
        self._marker_pub = node.create_publisher(ImageMarkerArray, cfg.markers_topic, 5)

        self._init_model(cfg)

    def detach_behaviour(self, node: Node):
        super().detach_behaviour(node)
        node.destroy_publisher(self._pred_pub)
        node.destroy_subscription(self._frames_sub)
        node.destroy_publisher(self._marker_pub)
        # ONNX Runtime has no python API for destroying a Session
        # So I assume the GC will auto-handle it (based on its C API)

    def on_params_change(self, node: Node, changes: dict):
        self.log.info(f"Config changed: {changes}.")
        if any(n in changes for n in ("frames_in_topic", "preds_out_topic")):
            self.log.info(f"Config change requires restart.")
            return True
        return False

    def step(self, delta: float):
        # Unused for this node
        return super().step(delta)

    def _init_model(self, cfg: YoloV5Cfg):
        self.log.info("Initializing ONNX...")
        self.log.info(f"Model Path: {cfg.model_path}")
        self.session = InferenceSession(cfg.model_path, providers=cfg.onnx_providers)
        # https://onnxruntime.ai/docs/api/python/api_summary.html#modelmetadata
        self.metadata = self.session.get_modelmeta()

        # these details were added by the YoloV5 toolkit
        model_details = self.metadata.custom_metadata_map
        self.log.info(f"Model Info: {self.metadata.custom_metadata_map}")
        if not model_details:
            self.log.warn("Model Metadata Empty! Assuming default COCO YOLO model...")
            self.stride = 64
            self.label_map = literal_eval(
                "['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']"
            )
        else:
            # TODO: imghw is fixed & should be read from metadata
            # imghw config option should be replaced with resizing behaviour
            # examples: contain, fill, stretch, tile (sliding window)
            self.stride = int(model_details["stride"])
            # YoloV5 has a security vulnerability where it doesnt store the label_map
            # as valid JSON, instead opting to use eval() to load it
            # Partially mitigated here by using literal_eval instead
            self.label_map: list = literal_eval(model_details["names"])

        # temporary until we figure out a better way to handle this
        self.class_include = ["person"]
        self.confidence = 0.5
        self.nms = 0.9
        self._classes = [
            self.label_map.index(c) for c in self.class_include if c in self.label_map
        ]
        self.log.info("ONNX initialized")

    def _forward(self, img):
        x = np.stack(
            (
                letterbox(
                    img, new_shape=self.cfg.img_hw, stride=self.stride, auto=False
                )[0],
            ),
            0,
        )  # NHWC, RGB, float32
        x = (x / 255).transpose(0, 3, 1, 2).astype(np.float32)  # NCHW, RGB

        y = self.session.run(
            [self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: x}
        )[
            0
        ]  # output #0: (N, CONCAT, 85)
        dets = non_max_suppression(y, self.confidence, self.nms, self._classes)[
            0
        ]  # [N * (D, 6)] XYXY, CONF, CLS; get only 0th image given batchsize=1
        dets[:, :4] = scale_coords(self.cfg.img_hw, img.shape, dets[:, :4])
        return dets

    def _on_input(self, msg: Image):
        if self._pred_pub.get_subscription_count() < 1:
            return
        if self.cfg.accept_compression:
            img = cv_bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
        else:
            img = cv_bridge.imgmsg_to_cv2(msg, "rgb8")
        if 0 in img.shape:
            return

        dets = self._forward(img).astype(float)  # ROS2 msgs are too type-sensitive

        self._pred_pub.publish(
            Pred2DArray(
                preds=[
                    Pred2D(
                        header=msg.header,
                        pred=BBox2D(is_norm=False, type=BBox2D.XYXY, box=det[:4]),
                        score=det[4],
                        label=self.label_map[int(det[5])],
                    )
                    for det in dets
                ]
            )
        )

        self._marker_pub.publish(
            ImageMarkerArray(
                markers=[
                    ImageMarker(
                        header=msg.header,
                        scale=1.0,
                        type=ImageMarker.POLYGON,
                        outline_color=ColorRGBA(r=1.0, a=1.0),
                        points=[
                            Point(x=det[0], y=det[1]),
                            Point(x=det[2], y=det[1]),
                            Point(x=det[2], y=det[3]),
                            Point(x=det[0], y=det[3]),
                        ],
                    )
                    for det in dets
                ]
            )
        )


def main(args=None):
    if __name__ == "__main__" and args is None:
        args = sys.argv

    rclpy.init(args=args)

    node = Node(NODE_NAME)

    cfg = YoloV5Cfg()
    YoloV5Predictor(node, cfg)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
