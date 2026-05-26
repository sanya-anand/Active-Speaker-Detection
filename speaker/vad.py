import webrtcvad
import wave


class VoiceActivityDetector:

    def __init__(self):

        self.vad = webrtcvad.Vad(2)

    def detect_speech(self, audio_path):

        wf = wave.open(audio_path, 'rb')

        sample_rate = wf.getframerate()

        frame_duration = 30
        frame_size = int(sample_rate * frame_duration / 1000) * 2

        speech_segments = []

        frame_index = 0

        while True:

            frame = wf.readframes(frame_size // 2)

            if len(frame) < frame_size:
                break

            is_speech = self.vad.is_speech(frame, sample_rate)

            start_time = frame_index * frame_duration / 1000
            end_time = start_time + frame_duration / 1000

            speech_segments.append({
                "start": start_time,
                "end": end_time,
                "speech": is_speech
            })

            frame_index += 1

        wf.close()

        return speech_segments