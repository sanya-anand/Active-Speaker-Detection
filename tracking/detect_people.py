from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model_path):

        self.model = YOLO(model_path)

    def detect(self, frame):

        results = self.model(
            frame,
            classes=[0],
            verbose=False
        )

        persons = []

        for result in results:

            boxes = result.boxes

            for box in boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                confidence = float(box.conf[0])

                persons.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence
                })

        return persons