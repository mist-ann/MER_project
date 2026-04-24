import os
import librosa
import numpy as np

def extract_co(audio_data, sr):
    cochleagram = librosa.cqt(audio_data, sr=sr)
    cochleagram_log = librosa.amplitude_to_db(np.abs(cochleagram))
    return cochleagram_log

def process_wav_file(input_file, output_folder):
    audio_data, sr = librosa.load(input_file, sr=None)

    start_time = 14
    end_time = 60
    step_size = 1

    start_frame = int(start_time * sr)
    end_frame = int(end_time * sr)

    co_list = []

    for i, start_frame in enumerate(range(start_frame, end_frame, int(step_size * sr))):
        end_frame = start_frame + int(step_size * sr)
        segmented_audio = audio_data[start_frame:end_frame]
        if len(segmented_audio) < int(0.5 * sr):
            segmented_audio = np.pad(segmented_audio, (0, int(0.5 * sr) - len(segmented_audio)))
        cochleagram = extract_co(segmented_audio,sr)
        co_list.append(cochleagram)

    co_array = np.array(co_list)
    output_file_path = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(input_file))[0]}.npy')
    np.save(output_file_path, co_array)

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    wav_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    count = 0
    for wav_file in wav_files:
        wav_file_path = os.path.join(input_folder, wav_file)
        process_wav_file(wav_file_path, output_folder)
        count+=1
        print(count)

input_folder = 'music/data_scut/data_scut_clean'  
output_folder = 'music/data_scut/cochlegram'

process_folder(input_folder, output_folder)


