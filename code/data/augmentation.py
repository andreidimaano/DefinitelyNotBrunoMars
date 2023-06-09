import numpy as np
import librosa

def time_stretch(data, factor):
  audio_stretched = librosa.effects.time_stretch(data, rate = factor)
  return audio_stretched

def pitch_shift(data, factor, sr=22050):
  return librosa.effects.pitch_shift(data, sr=sr, n_steps=factor)

def white_noise(data, factor): # needs testing
  w_noise = np.random.normal(0, data.std(), data.size)
  new_data = data + (factor * w_noise)
  return new_data
    
def harm_distort(data, factor=None): # needs testing
   # https://arxiv.org/pdf/1912.07756.pdf
   new_data = np.sin(np.sin(np.sin(np.sin(np.sin(2*np.pi*data)))))
   return new_data