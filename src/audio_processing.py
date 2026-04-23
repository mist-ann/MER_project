from pydub import AudioSegment


def convert_to_wav(input_file, output_file):
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format="wav")


def convert_folder_to_wav(input_folder, output_folder):
    import os

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".wav")
            convert_to_wav(input_file, output_file)
