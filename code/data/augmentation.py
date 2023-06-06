# %%
import soundfile as sf
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import librosa
import pandas as pd
import IPython.display as ipd
import os
import random

# %%
# time stretch issue:
# audio_stretched speeds up / slows down audio but resampled negates the effect
# the result is a pitch shifted audio at original speeds
# possible fix? : https://librosa.org/doc/main/generated/librosa.util.fix_length.html
def time_stretch(data, factor):
  audio_stretched = librosa.effects.time_stretch(data, rate = factor)
  # audio_stretched_resampled = librosa.resample(audio_stretched,  orig_sr = len(audio_stretched), target_sr = len(data))
  return audio_stretched #audio_stretched_resampled

def pitch_shift(data, factor, sr=22050):
  return librosa.effects.pitch_shift(data, sr=sr, n_steps=factor)

def white_noise(data, factor): # needs testing
  w_noise = np.random.normal(0, data.std(), data.size)
  # may need to replace self.sr with data.size
  new_data = data + (factor * w_noise)
  return new_data
    
def harm_distort(data, factor=None): # needs testing
   # https://arxiv.org/pdf/1912.07756.pdf
   new_data = np.sin(np.sin(np.sin(np.sin(np.sin(2*np.pi*data)))))
   return new_data

# %%
# spectrogram augments
# frequency mask
def freq_mask(spec, length = 4, zero = True):
  # copy spectrogram data (presever original, return new augmented)
  new_spec = np.copy(spec)
  
  # n_mels = 80 so length can range from 0-80 
  # I think frequency range depends on n_mels but could be wrong 
  # If wrong, then use percentages (see time masking)
  
  # optional : additional line to randomize mask length from 0 - (length-1)
  # mask = random.randrange(0, length)
  
  # get start and end indexes
  start = random.randrange(0, new_spec.shape[0]-length)
  end = start+length

  # if start and end are not valid indexes return spectrogram with no augments
  if start < 0 or start >= new_spec.shape[0]:
    return new_spec
  if end < 0 or end >=new_spec.shape[0]:
    return new_spec

  # replace the mask area with 0 or song mean
  if zero is True: 
    new_spec[:][start:end] = 0
  else :
    new_spec[:][start:end] = new_spec.mean()

  return new_spec


# time masking
def time_mask(spec, perc = 5, zero = True):
  # basically same proceedure as freq_mask
  new_spec = np.copy(spec)

  # optional : additional line to randomize mask percentage from 0 - (perc-1)
  # mask = random.randrange(0, perc)

  # using percentages because song lengths vary 
  # if lengths are maintained, revert back to using length (see freq_mask)
  length = int((perc/100)*new_spec.shape[1])

  # adjust length if result is 0
  if length == 0:
    length = 1 

  # get start and end indexes
  start = random.randrange(0, new_spec.shape[1]-length)
  end = start+length

  # if start and end are not valid indexes return spectrogram with no augments
  if start < 0 or start >= new_spec.shape[1]:
    return new_spec
  if end < 0 or end >=new_spec.shape[1]:
    return new_spec

  # replace the mask area with 0 or song mean
  if zero is True: 
    new_spec[:][:, start:end] = 0
  else :
    new_spec[:][:, start:end] = new_spec.mean()

  return new_spec



