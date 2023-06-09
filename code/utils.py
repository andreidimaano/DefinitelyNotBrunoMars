import soundfile as sf
from pydub import AudioSegment
import librosa
import librosa
from librosa.feature import mfcc
import numpy as np

def spectrogram_to_wav(spectrogram, mel_mean, mel_std, output_path, sr=22050):
    """
    Args:
        spectrogram (torch.Tensor): spectrogram to be converted
        mel_mean: Mean of spectrogram
        mel_std: std of spectrogram
        output_path: the audio will be saved to this path

    Returns:
        torch.Tensor: wav that represents the decoded spectrogram
    """
    spect_converted = spectrogram * mel_std + mel_mean
    reconstructed_audio = librosa.feature.inverse.mel_to_audio(librosa.db_to_power(spect_converted), sr=sr,  n_fft=1024, hop_length=256)


    # Save the reconstructed waveform as a .wav file
    if not output_path.endswith(".wav"):
      output_path += '.wav'
    sf.write(output_path, reconstructed_audio, sr)
    return spect_converted 

def wav_to_mp3(wav_path, output_path):
   audio = AudioSegment.from_wav(wav_path)
   audio.export(output_path, format="mp3")
   

def get_mfccs_of_mel_spectogram(mel_spectogram):
    mfccs = mfcc(
        S=mel_spectogram,
        n_mfcc=35,
        norm=None,
        y=None,
        sr=None,
        dct_type=2,
        lifter=0,
    )
    # according to "Mel-Cepstral Distance Measure for Objective Speech Quality Assessment" by R. Kubichek, the zeroth
    # coefficient is omitted
    # there are different variants of the Discrete Cosine Transform Type II, the one that librosa's MFCC uses is 2 times
    # bigger than the one we want to use (which appears in Kubicheks paper)
    mfccs = mfccs[1:] / 2
    return mfccs

def get_mcd(spec1, spec2):
    mfcc1 = get_mfccs_of_mel_spectogram(spec1)
    mfcc2 = get_mfccs_of_mel_spectogram(spec2)
    mdiff = mfcc1-mfcc2
    mdiffnorm = np.linalg.norm(mdiff, axis=0)
    mcd = np.mean(mdiffnorm)
    return mcd