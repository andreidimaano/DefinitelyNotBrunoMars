import soundfile as sf
from pydub import AudioSegment
import librosa

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
    reconstructed_audio = librosa.feature.inverse.mel_to_audio(librosa.db_to_power(spect_converted))


    # Save the reconstructed waveform as a .wav file
    if not output_path.endswith(".wav"):
      output_path += '.wav'
    sf.write(output_path, reconstructed_audio, sr)

def wav_to_mp3(wav_path, output_path):
   audio = AudioSegment.from_wav(wav_path)
   audio.export(output_path, format="mp3")
