import cv2
import os
import supervision as sv
from ultralytics import YOLOv10
from pathlib import Path


HOME = Path(__file__).resolve().parent.parent


class VideoObjectDetection:
    def __init__(self, video_path, output_dir="output"):
        # self.model = YOLOv10(model_path)
        # self.model1 = YOLOv10(model_path1)
        self.models = []
        self.video_path = video_path
        self.output_dir = output_dir
        self.cap = cv2.VideoCapture(self.video_path)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        os.makedirs(self.output_dir, exist_ok=True)

        if not self.cap.isOpened():
            raise ValueError(f"Error: Unable to open video at path: {self.video_path}")
        
    def appendModel(self,modelPath):
        self.models.append(YOLOv10(modelPath))

    def process_frame(self, frame):
        Rs = []
        for model in self.models:
            Rs.append(model(frame)[0])
        
        for results in Rs:
            detections = sv.Detections.from_ultralytics(results)
            annotated_image = self.bounding_box_annotator.annotate(
            scene=frame, detections=detections)
            annotated_image = self.label_annotator.annotate(
            scene=annotated_image, detections=detections)
        return annotated_image

    def run(self):
  
        print("Starting video processing...")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("End of video or error reading frame.")
                break

            annotated_image = self.process_frame(frame)

            cv2.imshow('Object Detection', annotated_image)

            # Handle user input to close
            key = cv2.waitKey(2)
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
    MODEL_PATH = HOME/"models/human.pt"
    MODEL_PATH1 = HOME/"models/zebra.pt"


    VIDEO_PATH = HOME/"video/wideo2.mp4"

    detector = VideoObjectDetection( video_path=VIDEO_PATH)
    detector.appendModel(MODEL_PATH)
    detector.appendModel(MODEL_PATH1)
    detector.run()
