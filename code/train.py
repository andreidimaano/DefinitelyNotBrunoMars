import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.data as data

# from maskgan.model import Generator, Discriminator
from maskgan.modelv3 import Generator, Discriminator
# from args.cycleGAN_train_arg_parser import CycleGANTrainArgParser
from data.VCDataset import VCDataset
import h5py
import time
import matplotlib.pyplot as plt
# from mask_cyclegan_vc.utils import decode_melspectrogram, get_mel_spectrogram_fig
# from logger.train_logger import TrainLogger
# from saver.model_saver import ModelSaver

"""
Trains MaskCycleGAN-VC as described in https://arxiv.org/pdf/2102.12841.pdf
Inspired by https://github.com/jackaduma/CycleGAN-VC2
"""

class MaskCycleGANVCTraining(object):
    """Trainer for MaskCycleGAN-VC
    """

    def __init__(self, load_model=False):
        """
        Args:
            args (Namespace): Program arguments from argparser
        """
        # Store args
        self.training_time = 0
        self.average_time = 0
        self.num_epochs = 10
        self.start_epoch = 1
        self.generator_lr = 1e-5
        self.discriminator_lr = 1e-4
        self.decay_after = 1e5
        self.stop_identity_after = 1e4
        self.mini_batch_size = 1
        self.cycle_loss_lambda = 10
        self.identity_loss_lambda = 5
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.epochs_per_save = 100
        self.epochs_per_plot = 10

        # Initialize MelGAN-Vocoder used to decode Mel-spectrograms
        # self.vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        self.sample_rate = 22050

        # h5_file = h5py.File('data/iu.h5', 'r')
        # dataset_a = h5_file['audio']
        # self.datasetA_spec = np.array(dataset_a)
        # h5_file.close()

        # h5_file = h5py.File('data/bruno.h5', 'r')
        # dataset_b = h5_file['audio']
        # self.datasetB_spec = np.array(dataset_b)
        # h5_file.close()
        # Initialize speakerA's dataset
        self.datasetA_spec = self.loadPickleFile("data/iu_spec.pickle")
        self.datasetA_raw = self.loadPickleFile("data/iu_raw.pickle")
        datasetA_norm_stats = np.load("data/iu_norm_stat.npz")
        self.datasetA_spec_mean = datasetA_norm_stats['mean']
        self.datasetA_std = datasetA_norm_stats['std']

        # Initialize speakerB's dataset
        self.datasetB_spec = self.loadPickleFile("data/bruno_mars_spec.pickle")
        self.datasetB_raw = self.loadPickleFile("data/bruno_mars_raw.pickle")
        dataset_B_norm_stats = np.load("data/bruno_mars_norm_stat.npz")
        self.dataset_B_spec_mean = dataset_B_norm_stats['mean']
        self.dataset_B_spec_std = dataset_B_norm_stats['std']

        # Compute lr decay rate
        self.n_samples = len(self.datasetA_spec)
        print(f'n_samples = {self.n_samples}')
        self.generator_lr_decay = self.generator_lr / \
            float(self.num_epochs * (self.n_samples // self.mini_batch_size))
        self.discriminator_lr_decay = self.discriminator_lr / \
            float(self.num_epochs * (self.n_samples // self.mini_batch_size))
        print(f'generator_lr_decay = {self.generator_lr_decay}')
        print(f'discriminator_lr_decay = {self.discriminator_lr_decay}')

        # Initialize Train Dataloader
        self.num_frames = 64 
        self.dataset = VCDataset(datasetA_spec=self.datasetA_spec,
                    datasetB_spec=self.datasetB_spec,
                    datasetA_raw=self.datasetA_raw,
                    datasetB_raw=self.datasetB_raw,
                    n_frames=64, 
                    max_mask_len=25)
        self.train_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                            batch_size=self.mini_batch_size,
                                                            shuffle=True,
                                                            drop_last=False)

        # Initialize Validation Dataloader (used to generate intermediate outputs)
        self.validation_dataset = VCDataset(datasetA_spec=self.datasetA_spec,
                                            datasetB_spec=self.datasetB_spec,
                                            n_frames=320,
                                            max_mask_len=32,
                                            valid=True)
        self.validation_dataloader = torch.utils.data.DataLoader(dataset=self.validation_dataset,
                                                                 batch_size=1,
                                                                 shuffle=False,
                                                                 drop_last=False)


        # Initialize Generators and Discriminators
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)
        # Discriminator to compute 2 step adversarial loss
        self.discriminator_A2 = Discriminator().to(self.device)
        # Discriminator to compute 2 step adversarial loss
        self.discriminator_B2 = Discriminator().to(self.device)
        
        if load_model:
            self.generator_A2B.load_state_dict(torch.load('model_checkpoint/GA2B.pth'))
            self.generator_B2A.load_state_dict(torch.load('model_checkpoint/GB2A.pth'))
            self.discriminator_A.load_state_dict(torch.load('model_checkpoint/DA.pth'))
            self.discriminator_B.load_state_dict(torch.load('model_checkpoint/DB.pth'))
            # Discriminator to compute 2 step adversarial loss
            self.discriminator_A2.load_state_dict(torch.load('model_checkpoint/DA2.pth'))
            # Discriminator to compute 2 step adversarial loss
            self.discriminator_B2.load_state_dict(torch.load('model_checkpoint/DB2.pth'))

        # Initialize Optimizers
        g_params = list(self.generator_A2B.parameters()) + \
            list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
            list(self.discriminator_B.parameters()) + \
            list(self.discriminator_A2.parameters()) + \
            list(self.discriminator_B2.parameters())
        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

    def adjust_lr_rate(self, optimizer, generator):
        """Decays learning rate.

        Args:
            optimizer (torch.optim): torch optimizer
            generator (bool): Whether to adjust generator lr.
        """
        if generator:
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0., self.discriminator_lr - self.discriminator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.discriminator_lr

    def reset_grad(self):
        """Sets gradients of the generators and discriminators to zero before backpropagation.
        """
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def loadPickleFile(self, fileName):
        """Loads a Pickle file.

        Args:
            fileName (str): pickle file path

        Returns:
            file object: The loaded pickle file object
        """
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def get_training_time(self):
        return self.training_time, self.average_time

    def get_generator_dsicriminator(self):
        # return self.generator_B2A, self.discriminator_A, self.discriminator_B, self.discriminator_A2, self.discriminator_B2
        return self.generator_A2B
    
    def train(self):
        """Implements the training loop for MaskCycleGAN-VC
        """
        j = 0
        dloss_arr = []
        gloss_arr = []
        start_time = time.time()
        gloss = []
        dloss = []
        for epoch in range(self.start_epoch, self.num_epochs + 1):
            for i, (real_A, mask_A, real_B, mask_B) in enumerate(self.train_dataloader):                
                if j % 100 == 0:
                    print(f'iteration {j} start')

                with torch.set_grad_enabled(True):
                    real_A = real_A.to(self.device, dtype=torch.float)
                    mask_A = mask_A.to(self.device, dtype=torch.float)
                    real_B = real_B.to(self.device, dtype=torch.float)
                    mask_B = mask_B.to(self.device, dtype=torch.float)

                    # ----------------
                    # Train Generator
                    # ----------------
                    self.generator_A2B.train()
                    self.generator_B2A.train()
                    self.discriminator_A.eval()
                    self.discriminator_B.eval()
                    self.discriminator_A2.eval()
                    self.discriminator_B2.eval()

                    # Generator Feed Forward
                    fake_B = self.generator_A2B(real_A, mask_A)
                    cycle_A = self.generator_B2A(fake_B, torch.ones_like(fake_B))
                    fake_A = self.generator_B2A(real_B, mask_B)
                    cycle_B = self.generator_A2B(fake_A, torch.ones_like(fake_A))
                    identity_A = self.generator_B2A(
                        real_A, torch.ones_like(real_A))
                    identity_B = self.generator_A2B(
                        real_B, torch.ones_like(real_B))
                    d_fake_A = self.discriminator_A(fake_A)
                    d_fake_B = self.discriminator_B(fake_B)

                    # For Two Step Adverserial Loss
                    d_fake_cycle_A = self.discriminator_A2(cycle_A)
                    d_fake_cycle_B = self.discriminator_B2(cycle_B)

                    # Generator Cycle Loss
                    cycleLoss = torch.mean(
                        torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

                    # Generator Identity Loss
                    identityLoss = torch.mean(
                        torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

                    # Generator Loss
                    g_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                    g_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

                    # Generator Two Step Adverserial Loss
                    generator_loss_A2B_2nd = torch.mean((1 - d_fake_cycle_B) ** 2)
                    generator_loss_B2A_2nd = torch.mean((1 - d_fake_cycle_A) ** 2)

                    # Total Generator Loss
                    g_loss = g_loss_A2B + g_loss_B2A + \
                        generator_loss_A2B_2nd + generator_loss_B2A_2nd + \
                        self.cycle_loss_lambda * cycleLoss + self.identity_loss_lambda * identityLoss
                    # NOTE GENERATOR LOSS IS HERE
                    gloss.append(g_loss.item())
                    if j % 100 == 0:
                        gloss_arr.append(np.log(g_loss.clone().cpu().detach().item()))
                    
                    if j % 1000 == 0:
                        print("generator loss:", gloss_arr)
                        plt.plot(gloss_arr)
                        plt.title('Loss over Training')
                        plt.xlabel('Iterations')
                        plt.ylabel('Loss')

                        

                    # Backprop for Generator
                    self.reset_grad()
                    g_loss.backward()
                    self.generator_optimizer.step()

                    # ----------------------
                    # Train Discriminator
                    # ----------------------
                    self.generator_A2B.eval()
                    self.generator_B2A.eval()
                    self.discriminator_A.train()
                    self.discriminator_B.train()
                    self.discriminator_A2.train()
                    self.discriminator_B2.train()

                    # Discriminator Feed Forward
                    d_real_A = self.discriminator_A(real_A)
                    d_real_B = self.discriminator_B(real_B)
                    d_real_A2 = self.discriminator_A2(real_A)
                    d_real_B2 = self.discriminator_B2(real_B)
                    generated_A = self.generator_B2A(real_B, mask_B)
                    d_fake_A = self.discriminator_A(generated_A)

                    # For Two Step Adverserial Loss A->B
                    cycled_B = self.generator_A2B(
                        generated_A, torch.ones_like(generated_A))
                    d_cycled_B = self.discriminator_B2(cycled_B)

                    generated_B = self.generator_A2B(real_A, mask_A)
                    d_fake_B = self.discriminator_B(generated_B)

                    # For Two Step Adverserial Loss B->A
                    cycled_A = self.generator_B2A(
                        generated_B, torch.ones_like(generated_B))
                    d_cycled_A = self.discriminator_A2(cycled_A)

                    # Loss Functions
                    d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
                    d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
                    d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

                    d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
                    d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                    d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                    # Two Step Adverserial Loss
                    d_loss_A_cycled = torch.mean((0 - d_cycled_A) ** 2)
                    d_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)
                    d_loss_A2_real = torch.mean((1 - d_real_A2) ** 2)
                    d_loss_B2_real = torch.mean((1 - d_real_B2) ** 2)
                    d_loss_A_2nd = (d_loss_A2_real + d_loss_A_cycled) / 2.0
                    d_loss_B_2nd = (d_loss_B2_real + d_loss_B_cycled) / 2.0

                    # Final Loss for discriminator with the Two Step Adverserial Loss
                    d_loss = (d_loss_A + d_loss_B) / 2.0 + \
                        (d_loss_A_2nd + d_loss_B_2nd) / 2.0

                    if j % 1000 == 0:
                        print("d loss:", dloss_arr)
                        plt.plot(dloss_arr)
                        # Show the plot
                        plt.show()
                    # Backprop for Discriminator
                    self.reset_grad()
                    d_loss.backward()
                    self.discriminator_optimizer.step()
                if j > self.decay_after:
                    self.adjust_lr_rate(
                        self.generator_optimizer, generator=True)
                    self.adjust_lr_rate(
                        self.generator_optimizer, generator=False)
                
                if j > self.stop_identity_after:
                    self.identity_loss_lambda = 0
                j += 1
            
            torch.save(self.generator_A2B .state_dict(), 'model_checkpoint/GA2B.pth')
            torch.save(self.generator_B2A .state_dict(), 'model_checkpoint/GB2A.pth')
            torch.save(self.discriminator_A.state_dict(), 'model_checkpoint/DA.pth')
            torch.save(self.discriminator_B.state_dict(), 'model_checkpoint/DB.pth')
            torch.save(self.discriminator_A2.state_dict(), 'model_checkpoint/DA2.pth')
            torch.save(self.discriminator_B2.state_dict(), 'model_checkpoint/DB2.pth')
            
            print(f'epoch {epoch} done')
        end_time = time.time()
        csv_data = {"Generator Loss" : gloss, "Discriminator Loss" : dloss}
        df = pd.DataFrame.from_dict(csv_data)
        df.to_csv("Losses", index=False)
        torch.save(self.generator_A2B.state_dict(), "genAtoB.pth")
        torch.save(self.generator_B2A.state_dict(), "genBtoA.pth")
        torch.save(self.discriminator_A.state_dict(), "disA.pth")
        torch.save(self.discriminator_B.state_dict(), "disB.pth")
        torch.save(self.discriminator_A2.state_dict(), "disA2.pth")
        torch.save(self.discriminator_B2.state_dict(), "disB2.pth")

        self.training_time = end_time - start_time
        self.average_time = self.training_time / (self.num_epochs - self.start_epoch)