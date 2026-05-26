class AudioVisualSynchronizer:

    def __init__(self):
        pass

    def find_active_speaker(
        self,
        frame_time,
        speaking_people,
        speech_segments
    ):

        audio_active = False

        for segment in speech_segments:

            if (
                segment["start"]
                <= frame_time
                <= segment["end"]
            ):

                audio_active = segment["speech"]
                break

        if not audio_active:
            return None

        if len(speaking_people) == 0:
            return None

        # Single visible speaker
        if len(speaking_people) == 1:
            return speaking_people[0]

        # Multiple speaking faces
        # choose strongest mouth movement later
        return speaking_people[0]