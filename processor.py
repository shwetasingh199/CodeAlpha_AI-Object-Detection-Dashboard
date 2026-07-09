import cv2
import tempfile
import os
import time

from detector import ObjectDetector


class VideoProcessor:

    def __init__(self, confidence=0.25):

        self.detector = ObjectDetector(
            confidence=confidence
        )

        # Create output directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)

    # ------------------------------------
    # IMAGE PROCESSING
    # ------------------------------------
    def process_image(self, image):

        results = self.detector.detect(image)

        annotated = results[0].plot()

        boxes = results[0].boxes

        class_names = self.detector.names()

        detected_objects = []

        class_count = {}

        for box in boxes:

            cls = int(box.cls[0])

            name = class_names[cls]

            detected_objects.append(name)

            class_count[name] = class_count.get(name, 0) + 1

        return annotated, detected_objects, class_count

    # ------------------------------------
    # VIDEO PROCESSING
    # ------------------------------------
    def process_video(self, video_file):

        temp = tempfile.NamedTemporaryFile(delete=False)

        temp.write(video_file.read())
        temp.close()

        cap = cv2.VideoCapture(temp.name)

        if not cap.isOpened():
            raise RuntimeError("Unable to open uploaded video.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)

        if fps_video == 0:
            fps_video = 30

        output_path = "outputs/output.mp4"

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps_video,
            (width, height)
        )

        total_objects = 0

        class_count = {}
        while True:

            success, frame = cap.read()

            if not success:
                break

            start = time.time()

            results = self.detector.detect(frame)
            boxes = results[0].boxes

            class_names = self.detector.names()

            for box in boxes:

                cls = int(box.cls[0])

                name = class_names[cls]

                class_count[name] = class_count.get(name, 0) + 1
                return output_path, class_count
            
            annotated = results[0].plot()

            object_count = len(results[0].boxes)

            total_objects += object_count

            fps = 1 / (time.time() - start)

            cv2.putText(
                annotated,
                f"Objects: {object_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            writer.write(annotated)

        cap.release()
        writer.release()

        return output_path, total_objects