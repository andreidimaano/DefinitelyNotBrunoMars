import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class MusicDataset(Dataset):
    def __init__(self, file, time_warp_factor transform=None):
        self.data = h5py.File(file, 'r')
        self.mfccs = self.data.get("mfccs").value
        self.x = self.song.reshape((-1,1,64,94))
        self.len = self.x.shape[0]
        self.time_warp_factor = time_warp_factor

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        spectrogram = self.x[index]
        warped_spectrogram = self.apply_time_warp(spectrogram, self.time_warp_factor)
        return warped_spectrogram
    
    def apply_time_warp(self, spectrogram, time_warp_factor):
        time_axis = np.arange(spectrogram.shape[1])
        time_warp_factor = np.random.uniform(0, self.time_warp_factor)
        warped_time_axis = np.linspace(0, time_axis[-1], int(time_axis[-1] * (1 + time_warp_factor)))
        warped_spectrogram = torch.tensor(np.interp(warped_time_axis, time_axis, spectrogram[0]), dtype=torch.float32).unsqueeze(0)
        return warped_spectrogram
