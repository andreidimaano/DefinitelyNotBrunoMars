import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np

class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
    
    def forward(self, x):
        return x * t.sigmoid(x)
    
class UpSample(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super(UpSample,self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=5, stride=1, padding=2, padding_mode='reflect')
        self.upscale = nn.PixelShuffle(upscale_factor=2)
        self.nonlinearity = GLU()
        self.normFunc = nn.InstanceNorm2d(num_features=output_channels//4, affine=True)

    def forward(self, x):
        conv = self.conv(x)
        upscale = self.upscale(conv)
        normalize = self.normFunc(upscale)
        finalOutput = self.nonlinearity(normalize)
        return finalOutput
    
class DownSampleD(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super(DownSampleD,self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, padding_mode='reflect')
        self.nonlinearity = GLU()
        self.normFunc = nn.InstanceNorm2d(num_features=output_channels, affine=True)
    def forward(self, x):
        conv = self.conv(x)
        normalize = self.normFunc(conv)
        finalOutput = self.nonlinearity(normalize)
        return finalOutput
    
class DownSampleG(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super(DownSampleG,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=5, stride=2, padding=2, padding_mode='reflect')
        #self.conv1NonLinear = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=5, stride=2, padding=2, padding_mode='reflect')
        self.normFunc = nn.InstanceNorm2d(num_features=output_channels, affine=True)
        self.nonlinearity = GLU()

    def forward(self, x):
        conv = self.conv1(x)
        normalize = self.normFunc(conv)
        finalOutput = self.nonlinearity(normalize)
        return finalOutput
    
class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels) -> None: #input image should be a 64x94 I THINK IM NOT SURE
        super(ResBlock, self).__init__() #kernel should be 3, stride is 1, padding is 1
        self.nonlinearity = GLU()
        #self.scalarGLU = nn.Conv1d(in_channels=input_channels, output_channels=output_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv1d(in_channels=output_channels, out_channels=input_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.normFuncScalar = nn.InstanceNorm1d(num_features=output_channels, affine=True)
        self.normFunc = nn.InstanceNorm1d(num_features=input_channels, affine=True)
    
    def forward(self, x):
        firstH = self.conv1(x)
        normFirstH = self.normFuncScalar(firstH)
        #gatesFirstH = self.conv1(x)
        #normGatesH = self.normFuncScalar(gatesFirstH)
        firstGLU = self.nonlinearity(firstH)
        finalOutput = self.conv2(firstGLU)
        return finalOutput + x 

class Generator(nn.Module):
    def __init__(self, features=(80, 64), res_in_channels=256) -> None:
        super(Generator, self).__init__()
        time_bins, channels = features
        self.singledimchannels = time_bins//4 * res_in_channels
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=res_in_channels//2, kernel_size=(5,15), stride=1, padding=(2,7))
        #self.conv1GLU = nn.Conv2d(in_channels=2, out_channels=res_in_channels//2, kernel_size=(5,15), stride=1, padding=(2,7))
        self.downsample1 = DownSampleG(input_channels=res_in_channels//2, output_channels=res_in_channels)
        self.downsample2 = DownSampleG(input_channels=res_in_channels, output_channels=res_in_channels)
        self.conv2to1 = nn.Conv1d(in_channels=self.singledimchannels, out_channels=res_in_channels, kernel_size=1, stride=1)
        self.singletfan = nn.InstanceNorm1d(num_features=res_in_channels, affine=True)
        for i in range(6):
            self.add_module(f"resblock{i+1}", ResBlock(input_channels=res_in_channels, output_channels=res_in_channels*2))
        self.conv1to2 = nn.Conv1d(in_channels=res_in_channels, out_channels=self.singledimchannels, kernel_size=1, stride=1)
        self.doubletfan = nn.InstanceNorm1d(num_features=self.singledimchannels, affine=True)
        self.upsample1 = UpSample(input_channels=res_in_channels, output_channels=res_in_channels*4)
        self.upsample2 = UpSample(input_channels=res_in_channels, output_channels=res_in_channels*2)
        self.final = nn.Conv2d(in_channels=res_in_channels//2, out_channels=1, kernel_size=(5,15), stride=1, padding=(2,7), padding_mode='reflect')
        self.nonlinearity = GLU()

    def forward(self, x, mask):
        x = t.stack((x*mask, mask), dim=1)
        firstConv = self.nonlinearity(self.conv1(x))
        down1 = self.downsample1(firstConv)
        down2 = self.downsample2(down1)

        down2reshape = down2.view(down2.size(0), self.singledimchannels, 1, -1)
        down2reshape = down2reshape.squeeze(2)

        flattenLayer = self.conv2to1(down2reshape)
        flattenLayer = self.singletfan(flattenLayer)
        resBlockTime = flattenLayer
        for i in range(6):
            resBlockTime = self.__getattr__(f"resblock{i+1}")(resBlockTime)
        convUpTime = self.conv1to2(resBlockTime)
        convUpTime = self.doubletfan(convUpTime)
        convUpReshape = convUpTime.unsqueeze(2)
        convUpReshape = convUpReshape.view(convUpReshape.size(0), 256, 20, -1)

        upsample1 = self.upsample1(convUpReshape)
        upsample2 = self.upsample2(upsample1)

        finalOutput = self.final(upsample2)
        finalOutput = finalOutput.squeeze(1)
        return finalOutput
    
class Discriminator(nn.Module):
    def __init__(self, features=(80, 64), res_in_channels=256) -> None:
        super(Discriminator, self).__init__()
        self.nonlinearity = GLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=res_in_channels//2, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.downsample1 = DownSampleD(input_channels=res_in_channels//2, output_channels=res_in_channels)
        self.downsample2 = DownSampleD(input_channels=res_in_channels, output_channels=res_in_channels*2)
        self.downsample3 = DownSampleD(input_channels=res_in_channels*2, output_channels=res_in_channels*4)
        self.final = nn.Conv2d(in_channels=res_in_channels*4, out_channels=1, kernel_size=(1,3), stride=1, padding=(0,1), padding_mode='reflect')


    def forward(self, x):
        x = x.unsqueeze(1)
        firstConv = self.conv1(x) #REMEMBER TO DO NONLINEARITY
        firstNonlinearity = self.nonlinearity(firstConv)
        downs1 = self.downsample1(firstNonlinearity)
        downs2 = self.downsample2(downs1)
        downs3 = self.downsample3(downs2)

        finalOutput = self.final(downs3)
        return t.sigmoid(finalOutput)



         