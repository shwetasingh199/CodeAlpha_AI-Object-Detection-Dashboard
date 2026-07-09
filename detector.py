"""
detector.py
---------------------------------------
YOLOv8 Object Detection Module
---------------------------------------
Loads the YOLOv8 model and performs object detection.
"""

from ultralytics import YOLO
import torch


class ObjectDetector:

    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence=0.25,
        iou=0.45
    ):

        self.confidence = confidence
        self.iou = iou

        self.model = YOLO(model_path)

        # Automatically use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            print(f"Loading model: {model_path}")
            print(f"Running on: {self.device.upper()}")

            self.model = YOLO(model_path)

            # Move model to device
            self.model.to(self.device)

            print("Model Loaded Successfully!")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model.\n{e}"
            )

    # -------------------------------------------------

    def detect(self, frame):

        """
        Perform object detection.

        Parameters
        ----------
        frame : ndarray
            Image/frame

        classes : list
            Optional list of class IDs.
            Example:
            [0] -> Person only
            [0,2,3] -> Person, Car, Motorcycle
        """

        results = self.model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=self.confidence,
        iou=self.iou,
        device=self.device,
        verbose=False
        )

        return results

    # -------------------------------------------------

    def names(self):
        return self.model.names