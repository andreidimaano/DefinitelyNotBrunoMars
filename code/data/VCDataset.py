"""
taken from https://github.com/GANtastic3/MaskCycleGAN-VC/blob/main/dataset/vc_dataset.py#L11
with modifications for data augmentations

Dataset class for voice conversion. Returns pairs of Mel-spectrograms from
two datasets as well as corresponding masks.
"""

from torch.utils.data.dataset import Dataset
from .augmentation import *
import torch
import numpy as np
import copy
import random


class VCDataset(Dataset):
    def __init__(self, datasetA_spec, datasetB_spec, datasetA_raw=None, datasetB_raw=None, n_frames=64, max_mask_len=25, valid=False, 
        TimeStretch=False, PitchShift=False, HarmDist=False, WhiteNoise=False, TimeMask=False, FreqMask=False):
        self.datasetA_spec = datasetA_spec
        self.datasetA_raw = datasetA_raw
        self.datasetB_spec = datasetB_spec
        self.datasetB_raw = datasetB_raw
        self.n_frames = n_frames
        self.valid = valid
        self.max_mask_len = max_mask_len
        self.TimeStretch = TimeStretch 
        self.PitchShift=PitchShift
        self.HarmDist=HarmDist
        self.WhiteNoise=WhiteNoise
        self.TimeMask=TimeMask
        self.FreqMask=FreqMask

    def __getitem__(self, index):
        datasetA_spec = self.datasetA_spec.copy()
        datasetA_raw = self.datasetA_raw.copy()
        datasetB_spec = self.datasetB_spec.copy()
        datasetB_raw = self.datasetB_raw.copy()
        n_frames = self.n_frames # where each training sample consisted of 64 randomly cropped frames
       #sr is 22050 
        # Augmentations
        # VoTrans, NoisyF0
        # Time Warping
        # Frequency Masking
        # Speed up
        # Pitch Shift
        
        # THESE ARE SPECTOGRAM AUGS
        # TimeMask
        # FrequencyMask
        # nothing, 1, 2, 3, 4, 5, 6, (1,2), (1,6), (5,6)

        if self.TimeStretch or self.PitchShift or self.HarmDist or self.WhiteNoise:
            #APPLY AUG FIRST AND THEN CONVERT WAV TO SPECTO
            for i in range(len(datasetA_raw)):
                if self.TimeStretch and random.randint(0,5) == 0:
                    datasetA_raw[i] = time_stretch(data=datasetA_raw[i], factor=3)
                if self.PitchShift and random.randint(0,5) == 0:
                    datasetA_raw[i] = pitch_shift(data=datasetA_raw[i], sr=22050, factor=.1)
                if self.WhiteNoise and random.randint(0,5) == 0:
                    datasetA_raw[i] = pitch_shift(data=datasetA_raw[i], factor=1.5)
                if self.HarmDist and random.randint(0,5) == 0:
                    datasetA_raw[i] = pitch_shift(data=datasetA_raw[i])

                melA_spec = librosa.feature.melspectrogram(y=datasetA_raw[i], sr=22050, n_fft=1024, hop_length=256, n_mels=80)
                logmelA_spec = librosa.power_to_db(melA_spec, ref=np.max)
                datasetA_spec[i] = logmelA_spec
                #if self.TimeMask:
                #    datasetA_spec[i] = time_mask(datasetA_spec[i])
                #if self.FreqMask:
                #    datasetA_spec[i] = freq_mask(datasetA_spec[i])


            for i in range(len(datasetB_raw)):
                if self.TimeStretch and random.randint(0,5) == 0:
                    datasetB_raw[i] = time_stretch(data=datasetB_raw[i], factor=3)
                if self.PitchShift and random.randint(0,5) == 0:
                    datasetB_raw[i] = pitch_shift(data=datasetB_raw[i], sr=22050, factor=.1)
                if self.WhiteNoise and random.randint(0,5) == 0:
                    datasetB_raw[i] = pitch_shift(data=datasetB_raw[i], factor=1.5)
                if self.HarmDist and random.randint(0,5) == 0:
                    datasetB_raw[i] = pitch_shift(data=datasetB_raw[i])

                melB_spec = librosa.feature.melspectrogram(y=datasetB_raw[i], sr=22050, n_fft=1024, hop_length=256, n_mels=80)
                logmelB_spec = librosa.power_to_db(melB_spec, ref=np.max)
                datasetB_spec[i] = logmelB_spec
                #if self.TimeMask:
                #    datasetB_spec[i] = time_mask(datasetB_spec[i])
                #if self.FreqMask:
                #    datasetB_spec[i] = freq_mask(datasetB_spec[i])    
        if self.TimeMask or self.FreqMask:
            for i in range(len(datasetA_spec)):
                if self.TimeMask and random.randint(0,5) == 0:
                    datasetA_spec[i] = time_mask(datasetA_spec[i])
                if self.FreqMask and random.randint(0,5) == 0:
                    datasetA_spec[i] = freq_mask(datasetA_spec[i])
            
            for i in range(len(datasetB_spec)):
                if self.TimeMask and random.randint(0,5) == 0:
                    datasetB_spec[i] = time_mask(datasetB_spec[i])
                if self.FreqMask and random.randint(0,5) == 0:
                    datasetB_spec[i] = freq_mask(datasetB_spec[i]) 

        if self.valid:
            if datasetB_spec is None:  # only return datasetA utterance
                return datasetA_spec[index]
            else:
                return datasetA_spec[index], datasetB_spec[index]

        self.length = min(len(datasetA_spec), len(datasetB_spec))
        num_samples = min(len(datasetA_spec), len(datasetB_spec))

        train_data_A_idx = np.arange(len(datasetA_spec))
        train_data_B_idx = np.arange(len(datasetB_spec))
        np.random.shuffle(train_data_A_idx)  # Why do we shuffle?
        np.random.shuffle(train_data_B_idx)
        train_data_A_idx_subset = train_data_A_idx[:num_samples]
        train_data_B_idx_subset = train_data_B_idx[:num_samples]

        

        train_data_A = list()
        train_mask_A = list()
        train_data_B = list()
        train_mask_B = list()

        # apply mask
        for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
            data_A = datasetA_spec[idx_A]
            frames_A_total = data_A.shape[1]
            #print(data_A.shape)
            assert frames_A_total >= n_frames
            start_A = np.random.randint(frames_A_total - n_frames + 1)
            end_A = start_A + n_frames
            mask_size_A = np.random.randint(0, self.max_mask_len)
            assert n_frames > mask_size_A
            mask_start_A = np.random.randint(0, n_frames - mask_size_A)
            mask_A = np.ones_like(data_A[:, start_A:end_A])
            mask_A[:, mask_start_A:mask_start_A + mask_size_A] = 0.
            train_data_A.append(data_A[:, start_A:end_A])
            train_mask_A.append(mask_A)

            data_B = datasetB_spec[idx_B]
            frames_B_total = data_B.shape[1]
            #print(data_B.shape)
            assert frames_B_total >= n_frames
            start_B = np.random.randint(frames_B_total - n_frames + 1)
            end_B = start_B + n_frames
            mask_size_B = np.random.randint(0, self.max_mask_len)
            assert n_frames > mask_size_B
            mask_start_B = np.random.randint(0, n_frames - mask_size_B)
            mask_B = np.ones_like(data_A[:, start_A:end_A])
            mask_B[:, mask_start_B:mask_start_B + mask_size_B] = 0.
            train_data_B.append(data_B[:, start_B:end_B])
            train_mask_B.append(mask_B)

        train_data_A = np.array(train_data_A)
        train_data_B = np.array(train_data_B)
        train_mask_A = np.array(train_mask_A)
        train_mask_B = np.array(train_mask_B)

        return train_data_A[index], train_mask_A[index],  train_data_B[index], train_mask_B[index]

    def __len__(self):
        if self.datasetB_spec is None:
            return len(self.datasetA_spec)
        else:
            return min(len(self.datasetA_spec), len(self.datasetB_spec))


if __name__ == '__main__':
    # Trivial test for dataset class
    trainA = np.random.randn(162, 24, 554)
    trainB = np.random.randn(158, 24, 554)
    dataset = VCDataset(trainA, trainB)
    trainLoader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=2,
                                              shuffle=True)
    for i, (A, mask_A, B, mask_B) in enumerate(trainLoader):
        print(A.shape, mask_B.shape, B.shape, mask_B.shape)
        assert A.shape == mask_B.shape == B.shape == mask_B.shape
        break
