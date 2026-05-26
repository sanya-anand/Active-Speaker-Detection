from insightface.app import FaceAnalysis
import cv2


class FaceEmbedder:

    def __init__(self):

        # Load InsightFace model
        self.app = FaceAnalysis(name='buffalo_l')

        # GPU
        self.app.prepare(ctx_id=0)

    def get_faces(self, frame):

        faces = self.app.get(frame)

        return faces

    def get_embedding(self, frame):

        faces = self.app.get(frame)

        if len(faces) == 0:
            return None, None

        # Take biggest face
        face = max(
            faces,
            key=lambda x: (
                (x.bbox[2] - x.bbox[0]) *
                (x.bbox[3] - x.bbox[1])
            )
        )

        embedding = face.embedding
        bbox = face.bbox.astype(int)

        return embedding, bbox