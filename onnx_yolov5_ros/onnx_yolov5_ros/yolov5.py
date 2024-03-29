from __future__ import annotations

import sys
from ast import literal_eval
from dataclasses import dataclass, field

import numpy as np
import rclpy
from cv_bridge import CvBridge
from foxglove_msgs.msg import ImageMarkerArray
from geometry_msgs.msg import Point
from nicefaces.msg import BBox2D, ObjDet2Ds, TrackData
from nicepynode import Job, JobCfg
from nicepynode.utils import (
    RT_PUB_PROFILE,
    RT_SUB_PROFILE,
    append_array,
    convert_bboxes,
    declare_parameters_from_dataclass,
    letterbox,
)
from onnxruntime import (
    ExecutionMode,
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)
from rclpy.node import Node
from ros2topic.api import get_msg_class
from sensor_msgs.msg import Image
from visualization_msgs.msg import ImageMarker

from onnx_yolov5_ros.processing import non_max_suppression, scale_coords

NODE_NAME = "yolov5_model"

cv_bridge = CvBridge()

# Tuning Guide: https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/docs/ONNX_Runtime_Perf_Tuning.md
# and https://onnxruntime.ai/docs/performance/tune-performance.html
SESS_OPTS = SessionOptions()
# opts.enable_profiling = True
SESS_OPTS.enable_mem_pattern = True  # is default
SESS_OPTS.enable_mem_reuse = True  # is default
SESS_OPTS.execution_mode = ExecutionMode.ORT_PARALLEL  # does nothing on CUDA
SESS_OPTS.intra_op_num_threads = 2  # does nothing on CUDA
SESS_OPTS.inter_op_num_threads = 2  # does nothing on CUDA
SESS_OPTS.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL  # is defaul
# CUDAExecutionProvider Options: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
PROVIDER_OPTS = [
    dict(
        device_id=0,
        gpu_mem_limit=2 * 1024 ** 3,
        arena_extend_strategy="kSameAsRequested",
        do_copy_in_default_stream=False,
        cudnn_conv_use_max_workspace=True,
        cudnn_conv1d_pad_to_nc1d=True,
        cudnn_conv_algo_search="EXHAUSTIVE",
        # enable_cuda_graph=True,
    )
]


@dataclass
class YoloV5Cfg(JobCfg):
    # TODO: Model selection API?
    model_path: str = "/models/yolov7-tiny.onnx"
    """Local path of model."""
    frames_in_topic: str = "~/frames_in"
    """Video frames to predict on."""
    preds_out_topic: str = "~/preds_out"
    """Output topic for predictions."""
    markers_out_topic: str = "~/bbox_markers"
    """Output topic for visualization markers."""
    onnx_providers: list[str] = field(default_factory=lambda: ["CUDAExecutionProvider"])
    """ONNX runtime providers."""
    # TODO: img_wh should be embedded within exported model metadata
    img_wh: tuple[int, int] = (640, 640)
    """Input resolution."""
    # NOTE: increasing score_threshold & lowering nms_threshold reduces lag
    score_threshold: float = 0.4
    """Minimum confidence level for filtering."""
    nms_threshold: float = 0.5
    """IoU threshold for non-maximum suppression."""
    # NOTE: filtering out classes reduces lag
    # TODO: class_exclude option
    class_include: list[str] = field(default_factory=lambda: ["person"])
    """Which classes to include, leave empty to include all."""
    # TODO: resizing behaviour option e.g. contain, fill, stretch, tile (sliding window)
    # TODO: config options for bbox output format such as:
    #  - points vs bbox
    #  - normalized or not
    #  - bbox type, XYXY vs XYWH vs CBOX
    assume_coco: bool = False
    """Assume given YOLO model is trained for COCO 80 classes."""
    assume_face: bool = False
    """For non-compliant YOLO face model from https://github.com/deepcam-cn/yolov5-face"""
    bbox_type: int = BBox2D.XYXY
    """Output type for bbox."""
    normalize_coords: bool = False
    """Whether to normalize bbox."""


@dataclass
class YoloV5Predictor(Job[YoloV5Cfg]):

    ini_cfg: YoloV5Cfg = field(default_factory=YoloV5Cfg)

    def attach_params(self, node: Node, cfg: YoloV5Cfg):
        super(YoloV5Predictor, self).attach_params(node, cfg)

        # onnx_providers & img_wh are hardcoded
        declare_parameters_from_dataclass(
            node, cfg, exclude_keys=["onnx_providers", "img_wh"]
        )

    def attach_behaviour(self, node: Node, cfg: YoloV5Cfg):
        super(YoloV5Predictor, self).attach_behaviour(node, cfg)

        self._init_model(cfg)

        # TODO: make this isomorphic realtime image subscriber a utility
        self.log.info(f"Waiting for publisher@{cfg.frames_in_topic}...")
        self._frames_sub = node.create_subscription(
            # blocks until image publisher is up!
            get_msg_class(node, cfg.frames_in_topic, blocking=True),
            cfg.frames_in_topic,
            self._on_input,
            RT_SUB_PROFILE,
        )
        self._pred_pub = node.create_publisher(
            ObjDet2Ds, cfg.preds_out_topic, RT_PUB_PROFILE
        )
        self._marker_pub = node.create_publisher(
            ImageMarkerArray, cfg.markers_out_topic, RT_PUB_PROFILE
        )
        self.log.info("Ready")

    def detach_behaviour(self, node: Node):
        super().detach_behaviour(node)
        node.destroy_subscription(self._frames_sub)
        node.destroy_publisher(self._pred_pub)
        node.destroy_publisher(self._marker_pub)
        # ONNX Runtime has no python API for destroying a Session
        # So I assume the GC will auto-handle it (based on its C API)

    def on_params_change(self, node: Node, changes: dict):
        self.log.info(f"Config changed: {changes}.")
        if not all(
            n
            in (
                "score_threshold",
                "nms_threshold",
                "class_include",
                "bbox_type",
                "normalize_bbox",
            )
            for n in changes
        ):
            self.log.info(f"Config change requires restart.")
            return True
        # force recalculation of included classes
        if "class_include" in changes:
            self._clsid_include = None
        return False

    @property
    def clsid_include(self):
        if self.label_map is None:
            self._clsid_include = []
        if self._clsid_include is None:
            self._clsid_include = [
                self.label_map.index(c)
                for c in self.cfg.class_include
                if c in self.label_map
            ]

        return self._clsid_include

    def _init_model(self, cfg: YoloV5Cfg):
        self.log.info("Initializing ONNX...")

        self.log.info(f"Model Path: {cfg.model_path}")
        self.session = InferenceSession(
            cfg.model_path,
            providers=cfg.onnx_providers,
            # performance gains measured to be negligable...
            sess_options=SESS_OPTS,
            provider_options=PROVIDER_OPTS,
        )
        # self.log.info(f"Options: {self.session.get_provider_options()}")
        # https://onnxruntime.ai/docs/api/python/api_summary.html#modelmetadata
        self.metadata = self.session.get_modelmeta()
        self._sess_out_name = self.session.get_outputs()[0].name
        self._sess_in_name = self.session.get_inputs()[0].name

        # these details were added by the YoloV5 toolkit
        model_details = self.metadata.custom_metadata_map
        self.log.info(f"Model Info: {self.metadata.custom_metadata_map}")

        # YoloV7 toolkit does not add model_details
        if not model_details:
            # TODO: THESE ARE HARDCODED ASSUMPTIONS
            self.log.warn("Model Metadata Empty!")
            self.stride = 32

            self.log.warn(
                f"Assumption settings: assume_coco={cfg.assume_coco}, assume_solo={cfg.assume_face}"
            )
            if cfg.assume_coco:
                self.label_map = literal_eval(
                    "['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']"
                )
            elif cfg.assume_face:
                self.label_map = None
            else:
                self.label_map = None

        else:
            self.stride = int(model_details["stride"])
            # bad design choice by YOLOv5 toolkit, we can only mitigate it...
            self.label_map: list = literal_eval(model_details["names"])

        self._clsid_include = None
        self.log.info("ONNX initialized")

    def _forward(self, img):
        # self.log.info(f"Shape: {img.shape}")
        x = letterbox(img, new_shape=self.cfg.img_wh, stride=self.stride, auto=False)[
            0  # (im, ratio, pad), take 0th element which is im
        ][None].astype(np.float32)
        # NHWC, RGB, float32
        x = (x / 255).transpose(0, 3, 1, 2)  # NCHW, RGB

        # output #0: (N, CONCAT, 85) (x, y, x, y, obj conf, ...80 classes conf)
        y = self.session.run([self._sess_out_name], {self._sess_in_name: x})[0]

        # for face models from https://github.com/deepcam-cn/yolov5-face
        # output #0: (N, CONCAT, 16) (x, y, x, y, obj conf, (x, y) * 5 face landmarks, 1 * class conf)
        # face models rn processed wrong as face landmarks interpreted as classes
        # hence disable using classes for tracking for face models (technically theres only 1 class anyways)
        if self.label_map is None and self.cfg.assume_face:
            y = y[..., np.r_[:5, -1]]

        dets = non_max_suppression(
            y,
            self.cfg.score_threshold,
            self.cfg.nms_threshold,
            self.clsid_include if len(self.clsid_include) else None,
        )[
            0
        ]  # [N * (D, 6)] XYXY, CONF, CLS; get only 0th image given batchsize=1
        dets[:, :4] = scale_coords(self.cfg.img_wh, img.shape[1::-1], dets[:, :4])
        return dets

    def _on_input(self, msg: Image):
        if (
            self._pred_pub.get_subscription_count()
            + self._marker_pub.get_subscription_count()
            < 1
        ):
            return

        infer_start = self.get_timestamp()

        if isinstance(msg, Image):
            img = cv_bridge.imgmsg_to_cv2(msg, "rgb8")
        else:
            img = cv_bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
        if 0 in img.shape:
            self.log.debug("Image has invalid shape!")
            return

        # (D, 6) XYXY, CONF, CLS
        dets = self._forward(img)

        infer_end = self.get_timestamp()

        if self._pred_pub.get_subscription_count() > 0:
            detsmsg = ObjDet2Ds(header=msg.header)
            detsmsg.profiling.infer_start_time = infer_start
            detsmsg.profiling.infer_end_time = infer_end

            detsmsg.boxes.header = msg.header
            detsmsg.boxes.type = BBox2D.XYXY
            detsmsg.boxes.is_norm = False

            append_array(detsmsg.boxes.a, dets[:, 0])
            append_array(detsmsg.boxes.b, dets[:, 1])
            append_array(detsmsg.boxes.c, dets[:, 2])
            append_array(detsmsg.boxes.d, dets[:, 3])

            if self.cfg.bbox_type != BBox2D.XYXY or self.cfg.normalize_coords:
                detsmsg.boxes = convert_bboxes(
                    detsmsg.boxes,
                    to_type=self.cfg.bbox_type,
                    normalize=self.cfg.normalize_coords,
                    img_wh=img.shape[1::-1],
                )

            append_array(detsmsg.scores, dets[:, 4])

            for d in dets:
                label = (
                    self.label_map[int(d[5])]
                    if not self.label_map is None
                    else str(int(d[5]))
                )

                detsmsg.labels.append(label)
                detsmsg.tracks.append(
                    TrackData(
                        label=label, x=d[[0, 2]], y=d[[1, 3]], scores=(d[4], d[4])
                    )
                )

            self._pred_pub.publish(detsmsg)

        if self._marker_pub.get_subscription_count() > 0:
            markersmsg = ImageMarkerArray()
            # must be float64 vs float32 for Point()...
            for det in dets.astype(float):
                marker = ImageMarker(header=msg.header)
                marker.scale = 1.0
                marker.type = ImageMarker.POLYGON
                marker.outline_color.r = 1.0
                marker.outline_color.a = 1.0
                marker.points = (
                    Point(x=det[0], y=det[1]),
                    Point(x=det[2], y=det[1]),
                    Point(x=det[2], y=det[3]),
                    Point(x=det[0], y=det[3]),
                )
                markersmsg.markers.append(marker)
            self._marker_pub.publish(markersmsg)


def main(args=None):
    if __name__ == "__main__" and args is None:
        args = sys.argv

    try:
        rclpy.init(args=args)

        node = Node(NODE_NAME)

        cfg = YoloV5Cfg()
        YoloV5Predictor(node, cfg)

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
