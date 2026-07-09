"""
tracker.py

Runs YOLO object detection with ByteTrack.
"""

class ObjectTracker:

    def __init__(self, model):

        self.model = model

    def track_objects(self, frame):

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        return results