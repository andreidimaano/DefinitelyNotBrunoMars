import os
import librosa
import librosa.display
import numpy as np
from pydub import AudioSegment
import h5py

def mp3_to_wav(src_dir, dest_dir):
    files = os.listdir(src_dir)
    for file in files:
        name, _ = os.path.splitext(file)
        dest = name + '.wav'
        audio = AudioSegment.from_mp3(file)
        audio.export(f'{dest_dir}/{dest}', format='wav')

def wav_to_spectrogram(src_dir, dest_dir):
    files = os.listdir(src_dir)
    h5_file = h5py.File(dest_dir, "w")
    h5_arr = []
    for file in files:
        audio, sr = librosa.load(f'{src_dir}/{file}', sr=16000)
        mfccs = librosa.feature.mfcc(audio,sr=sr)
        h5_arr.append(mfccs)
    h5_arr = np.array(h5_arr)
    h5_file.create_dataset("mfccs", data=h5_arr)
    h5_file.close()


wav_file = 'path/to/your/file.wav'
spectrogram = wav_to_spectrogram(wav_file)