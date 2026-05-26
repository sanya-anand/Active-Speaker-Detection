import cv2
from ultralytics import YOLO

from tracking.face_embedding import FaceEmbedder
from tracking.identity_manager import IdentityManager

from speaker.mouth_activity import MouthActivityDetector

from audio.extract_audio import extract_audio
from speaker.vad import VoiceActivityDetector
from speaker.sync_av import AudioVisualSynchronizer
from speaker.speaker_mapper import SpeakerMapper

# LOAD MODELS

model = YOLO("models/yolov8m.pt")

face_embedder = FaceEmbedder()

identity_manager = IdentityManager()

mouth_detector = MouthActivityDetector()



# VIDEO


cap = cv2.VideoCapture("input/sample2.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

print("VIDEO FPS:", fps)

# AUDIO EXTRACTION

extract_audio(
    "input/sample2.mp4",
    "audio.wav"
)

vad = VoiceActivityDetector()

speech_segments = vad.detect_speech("audio.wav")

sync = AudioVisualSynchronizer()

speaker_mapper = SpeakerMapper()

frame_count = 0

previous_mouth = {}
speaker_scores = {}

# MAIN LOOP

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    speaking_people = []


    # YOLO PERSON DETECTION


    results = model(frame, classes=[0], verbose=False)

    for result in results:

        boxes = result.boxes

        for box in boxes:

            conf = float(box.conf[0])

            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop person
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

        
            # FACE EMBEDDING
        

            embedding, face_bbox = face_embedder.get_embedding(person_crop)

            if embedding is None:
                continue

            person_id = identity_manager.assign_id(embedding)

        
            # FACE CROP
        

            fx1, fy1, fx2, fy2 = face_bbox

            face_crop = person_crop[fy1:fy2, fx1:fx2]

            if face_crop.size == 0:
                continue

        
            # MOUTH ACTIVITY
        

            processed_face, mouth_distance = mouth_detector.process_face(
                face_crop
            )

        
            # DYNAMIC MOUTH MOVEMENT
        

            is_speaking = False

            if person_id not in previous_mouth:

                previous_mouth[person_id] = mouth_distance

            movement = abs(
                mouth_distance -
                previous_mouth[person_id]
            )

            # Update memory
            previous_mouth[person_id] = mouth_distance

        
            # SPEAKING CONFIDENCE
        

            if person_id not in speaker_scores:

                speaker_scores[person_id] = 0

            # Increase score if mouth moves
            if movement > 0.8:

                speaker_scores[person_id] += 2

            else:

                speaker_scores[person_id] -= 1

            # Clamp values
            speaker_scores[person_id] = max(
                0,
                min(10, speaker_scores[person_id])
            )

            # Final speaking decision
            if speaker_scores[person_id] >= 3:

                is_speaking = True

            # Save speaking IDs
            if is_speaking:
                speaking_people.append(person_id)

            # Put processed face back
            person_crop[fy1:fy2, fx1:fx2] = processed_face

            # Put face back into frame
            frame[y1:y2, x1:x2] = person_crop

        
            # DRAW PERSON BOX
        

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            status = "Speaking" if is_speaking else "Silent"

            cv2.putText(
                frame,
                f"ID {person_id} - {status}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # AUDIO VISUAL SYNC


    frame_time = frame_count / fps

    active_speaker = sync.find_active_speaker(
        frame_time,
        speaking_people,
        speech_segments
    )

    if active_speaker is not None:

        speaker_mapper.add_result(
            frame_time,
            active_speaker
        )

    # SHOW FRAME


    cv2.imshow("Speaker Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

speaker_mapper.print_summary()

cap.release()

cv2.destroyAllWindows()