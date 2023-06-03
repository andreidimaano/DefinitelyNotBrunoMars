import pickle
import os
import librosa
import numpy as np

def audio_to_spectrogram(src_dir, artist):
    """
    ARGS
        src_dir: directory with audio files
        artist: .pickle directory for artist
    """
    spec_dest = artist + "_spec" + ".pickle"
    raw_dest = artist + "_spec" + ".pickle"

    mel_list = list()
    raw_list = list()
    
    i = 0
    for root, _, files in os.walk(src_dir, topdown=False):
        for file_name in files:
            if file_name.endswith('.flac') or file_name.endswith('.mp3'):
                try:
                    input_file = os.path.join(root, file_name)
                    audio, sr = librosa.load(input_file, sr=22050)
                    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
                    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                    if log_mel_spectrogram.shape[-1] >= 64:    # training sample consists of 64 randomly cropped frames
                        mel_list.append(log_mel_spectrogram)
                        raw_list.append(audio)
                    # larger nfft better for frequency resolution
                    # higher hop length for time resolution
                    if i % 50 == 0:
                        print(f'{i} done')
                    i+=1
                except AttributeError as e:
                    print(f"Error processing file: {file_name} - {e}")
                    continue

    mel_concatenated = np.concatenate(mel_list, axis=1)
    mel_mean = np.mean(mel_concatenated, axis=1, keepdims=True)
    mel_std = np.std(mel_concatenated, axis=1, keepdims=True) + 1e-9

    mel_normalized = list()
    for mel in mel_list:
        assert mel.shape[-1] >= 64, f"Mel spectogram length must be greater than 64 frames, but was {mel.shape[-1]}"
        app = (mel - mel_mean) / mel_std
        mel_normalized.append(app)

    with open(spec_dest, 'wb') as file:
        pickle.dump(mel_normalized, file)

    with open(raw_dest, 'wb') as file:
        pickle.dump(raw_list, file)
        
    np.savez(f"{artist}_norm_stat.npz",
             mean=mel_mean,
             std=mel_std)
    
    return mel_normalized