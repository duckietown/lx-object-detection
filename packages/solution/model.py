#!/usr/bin/env python3

import time
import numpy as np
import cv2
from dt_computer_vision.camera.types import Pixel

from duckietown_messages.actuators.differential_pwm import DifferentialPWM
from solution.config import CONF_THRESHOLD, STOP_DISTANCE, FORWARD_PWM, DEBUG


class MLModel:
    def __init__(self, backend):
        print(f"Initializing MLModel (backend: {backend.__class__.__name__})")
        self.backend = backend
        self.net_h = backend.net_h
        self.net_w = backend.net_w
        self.in_dtype = backend.in_dtype
        self.conf_threshold = CONF_THRESHOLD
        self.stop_distance_m = STOP_DISTANCE
        self.ground_projector = None

    def _run_detector(self, img_bgr):
        x = self._preprocess(img_bgr)
        t0 = time.perf_counter()
        raw = self.backend.infer(x)
        if DEBUG:
            print(f"inference: {(time.perf_counter() - t0) * 1000:.1f} ms")
        return self._postprocess(raw)

    def _postprocess(self, raw: np.ndarray, iou_thr: float = 0.45) -> np.ndarray:
        """Normalize the raw model output to [N, 6] (x1, y1, x2, y2, score, class_id)."""
        return self._apply_nms(raw, iou_thr)

    def _apply_nms(self, raw: np.ndarray, iou_thr: float = 0.45) -> np.ndarray:
        """
        Apply Non-Maximum Suppression to the raw YOLO anchor output.
        raw: [4+C, anchors] - once transposed to [anchors, 4+C] the columns are: cx, cy, w, h, score_class0, ...
             (pixel coordinates in input image space)
        Returns [M, 6]  (x1, y1, x2, y2, score, class_id).
        """

        # TODO 1. Transpose raw to [anchors, 4+C]
        #         Extract cx, cy, w, h (columns 0-3) and class scores (columns 4+).
        #         Hint: class_scores.argmax(axis=1) gives the most-likely class per anchor,
        #               index into the scores array to get the winning score.
        #      2. Filter anchors below CONF_THRESHOLD
        #         If nothing passes the threshold, return np.zeros((0, 6), np.float32).
        #      3. Convert center format to corner format:
        #         cx/cy/w/h to x1/y1/x2/y2
        #      4. Run cv2.dnn.NMSBoxes to suppress overlapping boxes
        #         It expects boxes as a list of [x1, y1, w, h] (NOT x2/y2).
        #         If no boxes survive, return np.zeros((0, 6), np.float32).
        #      5. Assemble the surviving detections into an [M, 6] array:
        #         columns: x1, y1, x1+w, y1+h, score, class_id

        return np.zeros((0, 6), np.float32)

    def _should_stop(self, detections: np.ndarray): 
        
        stop = False

        for x1, y1, x2, y2, score, _ in detections:

            print(f"Detection: {x1}-{x2}, {y1}-{y2}, {score}")
            # TODO we don't want to consider detections with confidence (score) below CONF_THRESHOLD (a value you should set in config.py)


            # TODO we want to stop if there is a duckie closer than STOP_DISTANCE away
            # To calculate if the duckie is too close we need to convert the pixel coordinates to 
            # world coordinates. To do so you can use the `self.ground_projector` object which has
            # loaded the camera extrinsic calibration
            # Specifically, if you want to project an object of type `pix = Pixel(x=u, y=v)` to a ground plane
            # point, you can first convert it to a vector (`vec = self.ground_projector.camera.pixel2vector(pix)`) and
            # then you can intersect that vector with the ground plane (`self.ground_projector.vector2ground(vec)`). 
            # That will be the point on the ground plane corresponding to the input pixel. 

        return stop

    def _preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]

        if h != self.net_h or w != self.net_w:
            raise ValueError(
                f"Image size {h}x{w} does not match ONNX! Expected {self.net_h}x{self.net_w}"
            )

        img = img_bgr[:, :, ::-1].astype(self.in_dtype) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        return img

    def set_ground_projector(self, gp):
        self.ground_projector = gp

    def get_wheel_velocities_from_image(self, img: np.ndarray):
        try:
            detections = self._run_detector(img)
        except Exception as e:
            print(f"Inference error: {e}")
            return [DifferentialPWM(left=0.0, right=0.0), None]
        if self._should_stop(detections):
            return [DifferentialPWM(left=0.0, right=0.0), detections]
        else:
            return [DifferentialPWM(left=FORWARD_PWM, right=FORWARD_PWM), detections]
