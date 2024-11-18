import cv2
import os
import supervision as sv
from ultralytics import YOLOv10


class VideoObjectDetection:
    def __init__(self, model_path, video_path, output_dir="output"):
        self.model = YOLOv10(model_path)
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(self.video_path)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.cap.isOpened():
            raise ValueError(f"Error: Unable to open video at path: {self.video_path}")

    def process_frame(self, frame):
        """
        Process a single frame: run detection, annotate with bounding boxes and labels.
        """
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        annotated_image = self.bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        annotated_image = self.label_annotator.annotate(
            scene=annotated_image, detections=detections)

        return annotated_image

    def run(self):
        """
        Main method to process the video frame by frame.
        """
        print("Starting video processing...")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("End of video or error reading frame.")
                break

            annotated_image = self.process_frame(frame)

            cv2.imshow('Object Detection', annotated_image)

            # Handle user input to close
            key = cv2.waitKey(1)
            if key % 256 == 27:  # ESC key
                print("CLOSING")
                break

        self.cleanup()

    def cleanup(self):
        """
        Release resources and cleanup.
        """
        self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

#tutaj dobre sciezki!
if __name__ == "__main__":
    MODEL_PATH = r'C:\\Users\\Stasiu\\Desktop\\crosswalks\\CrosswalksAI\\models\\human.pt'
    VIDEO_PATH = r'C:\\Users\\Stasiu\\Desktop\\crosswalks\\CrosswalksAI\\video\\wideo2.mp4'

    detector = VideoObjectDetection(model_path=MODEL_PATH, video_path=VIDEO_PATH)
    detector.run()
