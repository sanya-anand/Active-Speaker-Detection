import subprocess


def extract_audio(video_path, output_audio):

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        output_audio
    ]

    subprocess.run(command)
