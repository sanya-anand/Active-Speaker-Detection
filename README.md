# Active-Speaker-Detection
Real-time active speaker detection using audio-visual synchronization, face tracking, and voice activity analysis.

## Features

- Person detection using YOLOv8
- Face embedding and identity tracking
- Mouth activity detection
- Voice Activity Detection (VAD)
- Audio-video synchronization
- Active speaker identification

## Project Structure

```text
audio_processing/
    extract_audio.py

tracking/
    detect_people.py
    face_embedding.py
    identity_manager.py
    track_people.py

speaker/
    mouth_activity.py
    speaker_mapper.py
    sync_av.py
    vad.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```
