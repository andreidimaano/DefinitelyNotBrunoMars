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
import random


class VCDataset(Dataset):
    def __init__(self, datasetA_spec, datasetB_spec, datasetA_aug=None, datasetB_aug=None, aug_list = None, n_frames=64, max_mask_len=25, valid=False, 
        freq_mask=False):
        self.datasetA_spec = datasetA_spec
        self.datasetA_aug = datasetA_aug
        self.datasetB_spec = datasetB_spec
        self.datasetB_aug = datasetB_aug
        self.n_frames = n_frames
        self.valid = valid
        self.max_mask_len = max_mask_len
        self.apply_freq_mask=freq_mask
        self.aug_list = aug_list

    def __getitem__(self, index):
        datasetA_spec = self.datasetA_spec
        datasetB_spec = self.datasetB_spec
        n_frames = self.n_frames # where each training sample consisted of 64 randomly cropped frames
        if self.valid:
            if datasetB_spec is None:  # only return datasetA utterance
                return datasetA_spec[index]
            else:
                return datasetA_spec[index], datasetB_spec[index]
        datasetA_spec = datasetA_spec.copy()
        datasetA_spec = datasetB_spec.copy()
        datasetA = None
        datasetB = None

        if self.datasetA_aug and self.datasetB_aug:
            aug_a = 0 if (not self.aug_list or len(self.aug_list) == 1) else np.random.randint(0, len(self.aug_list))
            aug_b = 0 if (not self.aug_list or len(self.aug_list) == 1) else np.random.randint(0, len(self.aug_list))
            datasetA = self.datasetA_aug[aug_a] if np.random.rand() <= 0.5 else datasetA_spec
            datasetB = self.datasetB_aug[aug_b] if np.random.rand() <= 0.5 else datasetB_spec
        else:
            datasetA = datasetA_spec
            datasetB = datasetB_spec

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
            data_A = datasetA[idx_A]
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

            data_B = datasetB[idx_B]
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
        
        # apply freq mask here
        if self.apply_freq_mask and np.random.rand() <= 0.5:
            train_data_A = self.freq_mask(train_data_A)
            train_data_B = self.freq_mask(train_data_B)
            train_mask_A = self.freq_mask(train_mask_A)
            train_mask_B = self.freq_mask(train_mask_B)
        
        return train_data_A[index], train_mask_A[index],  train_data_B[index], train_mask_B[index]
    def freq_mask(self, spec, length=4):
        start = random.randrange(0, spec.shape[0]-length)
        end = start+length

        # if start and end are not valid indexes return spectrogram with no augments
        if start < 0 or start >= spec.shape[0] or end < 0 or end >=spec.shape[0]:
            return spec
        
        spec[:][start:end] = spec.mean()
            
        return spec
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
