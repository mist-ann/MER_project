import os
from pydub import AudioSegment

folder_path = '/home/mist/Documents/datasets/MusicRawData'

for filename in os.listdir(folder_path):
    if filename.endswith(".mp3"):
        mp3_path = os.path.join(folder_path, filename)
        wav_path = os.path.join(folder_path, filename.replace('.mp3', '.wav'))
        
        print(mp3_path)
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        
        os.remove(mp3_path)
