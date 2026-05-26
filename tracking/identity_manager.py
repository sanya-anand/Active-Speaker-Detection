import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class IdentityManager:

    def __init__(self, similarity_threshold=0.45):

        # {
        #   1: [emb1, emb2, emb3],
        #   2: [emb1, emb2]
        # }

        self.known_embeddings = {}

        self.next_id = 1

        self.similarity_threshold = similarity_threshold

    def assign_id(self, embedding):

        # FIRST PERSON
        if len(self.known_embeddings) == 0:

            person_id = self.next_id

            self.known_embeddings[person_id] = [embedding]

            self.next_id += 1

            return person_id

        best_similarity = -1
        best_id = None

        # Compare against ALL stored embeddings
        for person_id, saved_embeddings in self.known_embeddings.items():

            similarities = []

            for saved_embedding in saved_embeddings:

                similarity = cosine_similarity(
                    [embedding],
                    [saved_embedding]
                )[0][0]

                similarities.append(similarity)

            avg_similarity = np.mean(similarities)

            if avg_similarity > best_similarity:

                best_similarity = avg_similarity
                best_id = person_id

        # SAME PERSON
        if best_similarity > self.similarity_threshold:

            # Store new embedding for memory
            self.known_embeddings[best_id].append(embedding)

            # Prevent infinite growth
            if len(self.known_embeddings[best_id]) > 30:
                self.known_embeddings[best_id].pop(0)

            return best_id

        # NEW PERSON
        new_id = self.next_id

        self.known_embeddings[new_id] = [embedding]

        self.next_id += 1

        return new_id