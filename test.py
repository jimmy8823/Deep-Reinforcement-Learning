import airsim
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

input_images_ch = 2
batch_size = 16

class D3QN(nn.Module):
    def __init__(self,):
        super().__init__()
        # conv 1
        self.conv1 = nn.Conv2d(input_images_ch,16,kernel_size=5,stride=2)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # conv2
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=2)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        #conv3
        self.conv3 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # result shape[16,32,7,5]
        #FC
        self.fc1 = nn.Linear(17920,128)
        self.fc2 = nn.Linear(128,13)
    def forward(self,x):
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.batch3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = torch.flatten(out)

        out = self.fc1(out)
        out = self.fc2(out)
        return out

def main():
    q_value = torch.tensor([10,100,1000,10000,100000])
    output = torch.argmax(q_value).item()
    print(output)
main()