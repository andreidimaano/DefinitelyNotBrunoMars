import pickle
from tqdm import tqdm

import torch
import torch.utils.data as data
import torchaudio
import pickle
import numpy as np
from maskgan.modelv3 import Generator

from data.VCDataset import VCDataset
from utils import spectrogram_to_wav

class MaskCycleGANVCTesting(object):
    """Tester for MaskCycleGAN-VC
    """
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.generator = Generator().to(self.device)
        self.generator.load_state_dict(torch.load('model_checkpoint/GA2B.pth'))
        
        
        self.dataset_A = self.loadPickleFile("data/iu_test_spec.pickle")
        dataset_A_norm_stats = np.load("data/iu_test_norm_stat.npz")
        self.dataset_A_mean = dataset_A_norm_stats['mean']
        self.dataset_A_std = dataset_A_norm_stats['std']

        
        dataset_B_norm_stats = np.load("data/bruno_mars_test_norm_stat.npz")
        self.dataset_B_mean = dataset_B_norm_stats['mean']
        self.dataset_B_std = dataset_B_norm_stats['std']
        
        self.dataset = VCDataset(datasetA_spec=self.dataset_A,
                                 datasetB_spec=None,
                                 valid=True)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           drop_last=False)
    
    def loadPickleFile(self, fileName):
        """Loads a Pickle file.

        Args:
            fileName (str): pickle file path

        Returns:
            file object: The loaded pickle file object
        """
        with open(fileName, 'rb') as f:
            return pickle.load(f)
        
    def test(self):
        for i, sample in enumerate(tqdm(self.test_dataloader)):
            real_A = sample[:,:,:2000]
            real_A = real_A.to(self.device, dtype=torch.float)
            # print(real_A.shape)
            fake_B = self.generator(real_A, torch.ones_like(real_A))

            spectrogram_to_wav(fake_B[0].detach(
            ).cpu().numpy(), self.dataset_B_mean, self.dataset_B_std, f'outputs/learned_weights/fakeB_{i}.wav', sr=22050)
            
            spectrogram_to_wav(real_A[0].detach(
            ).cpu().numpy(), self.dataset_A_mean, self.dataset_A_std, f'outputs/learned_weights/realA_{i}.wav', sr=22050)