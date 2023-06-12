import pickle
from tqdm import tqdm

import torch
import pickle
import numpy as np
from maskgan.modelv3 import Generator

from data.VCDataset import VCDataset
from utils import spectrogram_to_wav, get_mcd
import os

class MaskCycleGANVCTesting(object):
    """Tester for MaskCycleGAN-VC
    """
    def __init__(self, a_prefix, b_prefix, load_id):
        #  a_prefix, b_prefix, train_id, load_id = None, load_model=False)

        self.load_id = load_id
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.generator = Generator().to(self.device)
        self.generator.load_state_dict(torch.load(f'model_checkpoint/{load_id}/GA2B.pth'))
        
        
        self.dataset_A = self.loadPickleFile(f'data/testing_data/{a_prefix}_test_spec.pickle')
        dataset_A_norm_stats = np.load(f'data/testing_data/{a_prefix}_test_norm_stat.npz')
        self.dataset_A_mean = dataset_A_norm_stats['mean']
        self.dataset_A_std = dataset_A_norm_stats['std']

        
        dataset_B_norm_stats = np.load(f'data/testing_data/{b_prefix}_test_norm_stat.npz')
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
        """
        outputs to predictions to .wav for subjective scoring
        calculates mcd scores for objective scoring
        """
        
        # Specify the path to the directory you want to create
        directory_path = f'outputs/generated_audio/{self.load_id+1}'

        # Check if the directory already exists
        if not os.path.exists(directory_path):
            # Create the directory
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        mcds = list()
        for i, sample in enumerate(tqdm(self.test_dataloader)):
            # converts full if small sample else 10 second sample
            real_A = sample[:,:,:5000]
            real_A = real_A.to(self.device, dtype=torch.float)
            fake_B = self.generator(real_A, torch.ones_like(real_A))
            fake_B_np = fake_B[0].detach().cpu().numpy()
            real_A_np = real_A[0].detach().cpu().numpy()
            spectrogram_to_wav(fake_B_np, self.dataset_B_mean, self.dataset_B_std, f'outputs/generated_audio/{self.load_id + 1}/fakeB_{i}.wav', sr=22050)
            
            spectrogram_to_wav(real_A_np, self.dataset_A_mean, self.dataset_A_std, f'outputs/generated_audio/{self.load_id + 1}/realA_{i}.wav', sr=22050)
            
            mcds.append(get_mcd(fake_B_np, real_A_np))
        return np.array(mcds)