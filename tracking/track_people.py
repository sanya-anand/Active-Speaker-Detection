import cv2

from tracking.detect_people import PersonDetector
from tracking.face_embedding import FaceEmbedder
from tracking.identity_manager import IdentityManager


VIDEO_PATH = "input/sample2.mp4"

# Initialize modules
detector = PersonDetector("models/yolov8m.pt")

embedder = FaceEmbedder()

identity_manager = IdentityManager()


cap = cv2.VideoCapture(VIDEO_PATH)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    persons = detector.detect(frame)

    for person in persons:

        x1, y1, x2, y2 = person["bbox"]

        # Crop person
        person_crop = frame[y1:y2, x1:x2]

        # Detect face inside crop
        faces = embedder.get_faces(person_crop)

        if len(faces) == 0:
            continue

        # Use first detected face
        face = faces[0]

        embedding = face.embedding

        # Assign stable ID
        person_id = identity_manager.assign_id(
            embedding
        )

        # Draw bbox
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"ID {person_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()