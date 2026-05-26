import cv2
import mediapipe as mp
import math


class MouthActivityDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )

        # Upper lip
        self.upper_lip = 13

        # Lower lip
        self.lower_lip = 14

    def process_face(self, face_crop):

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        mouth_distance = 0

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            h, w, _ = face_crop.shape

            upper = face_landmarks.landmark[self.upper_lip]
            lower = face_landmarks.landmark[self.lower_lip]

            x1 = int(upper.x * w)
            y1 = int(upper.y * h)

            x2 = int(lower.x * w)
            y2 = int(lower.y * h)

            # Draw points
            cv2.circle(face_crop, (x1, y1), 3, (0,255,0), -1)
            cv2.circle(face_crop, (x2, y2), 3, (0,0,255), -1)

            # Mouth opening distance
            mouth_distance = math.dist((x1, y1), (x2, y2))

        return face_crop, mouth_distance