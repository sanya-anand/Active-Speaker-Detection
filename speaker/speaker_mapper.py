class SpeakerMapper:

    def __init__(self):

        self.results = []

    def add_result(
        self,
        frame_time,
        speaker_id
    ):

        self.results.append({
            "time": frame_time,
            "speaker": speaker_id
        })

    def print_summary(self):

        print("\nFINAL SPEAKER TIMELINE\n")

        for item in self.results:

            print(
                f"Time: {item['time']:.2f}s "
                f"-> Speaker ID: {item['speaker']}"
            )