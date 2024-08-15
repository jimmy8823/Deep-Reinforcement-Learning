import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)
    
class Encoder(nn.Module):
    def __init__(self, image_channel=2, hidden_dim=5120, latent_dim=128):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            Conv2dSame(image_channel,32, (5, 5), stride=(2, 2)),
            nn.MaxPool2d(3,(2,2)),
            ResidualBlock(32,32),
            ResidualBlock(32,64),
            ResidualBlock(64,128),
            #Flatten(),
            #nn.LeakyReLU()
        )
        """
            nn.Conv2d(image_channel, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        """
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fcmean = nn.Linear(256, latent_dim)
        self.fclogvar = nn.Linear(256, latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                #m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                m.weight.data = nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

    def forward(self, x):
        h = self.cnn(x) #
        #o = self.fc1(h) # 128
        #mean = self.fcmean(o)
        #logvar = self.fclogvar(o)
        #print(concat)
        return h #mean, logvar
    
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels ,kernel_size = 3, stride = 1, downsample = None ,conv1_pad=0 ,conv2_pad=0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        Conv2dSame(in_channels, out_channels, kernel_size = kernel_size, stride=(2, 2)),
                        )
        self.conv2 = nn.Sequential(
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        Conv2dSame(out_channels, out_channels, kernel_size = kernel_size, stride = 1)
                        )
        self.skip = Conv2dSame(in_channels,out_channels,1,[2,2])
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skipconnect = self.skip(x)
        out += skipconnect
        return out

class ResidualBlock_fix_pad(nn.Module):

    def __init__(self, in_channels, out_channels ,kernel_size = 3, stride = 1, downsample = None ,conv1_pad=0 ,conv2_pad=0,skip_pad=0):
        super(ResidualBlock_fix_pad, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.BatchNorm2d(in_channels),
                        nn.ReLU(),
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride=(2, 2),padding=conv1_pad),
                        )
        self.conv2 = nn.Sequential(
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = 1,padding=conv2_pad)
                        )
        self.skip = nn.Conv2d(in_channels,out_channels,1,[2,2],padding=skip_pad)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skipconnect = self.skip(x)
        out += skipconnect
        return out
    
class Conv2dSame(nn.Conv2d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.shape[2],x.shape[3]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    
x = torch.randn((1,2,144,256))
encoder = Encoder(2,5120,128)
y = encoder(x)

cnn = nn.Sequential(
    nn.Conv2d(2,32,5,2,(2,2)),
    nn.MaxPool2d(3,(2,2)),
    ResidualBlock_fix_pad(32,32,conv1_pad=(1,1),conv2_pad=(1,1),skip_pad=(0,0)), # output 1 32 18 32
    ResidualBlock_fix_pad(32,64,conv1_pad=(1,1),conv2_pad=(1,1),skip_pad=(0,0)), # output 1 64 9 16
    ResidualBlock_fix_pad(64,128,conv1_pad=(1,1),conv2_pad=(1,1),skip_pad=(0,0)) # output 1 128 5 8
)

y2 = cnn(x)

x3 = torch.randn((1,32,9,16))
conv = Conv2dSame(32, 32, kernel_size = 1, stride=2)
y3 = conv(x3)

conv2 = nn.Conv2d(32, 32, kernel_size = 1, stride=2,padding=(0,0))
y4 = conv2(x3)
print("same pad :",y.shape)
print("fix :",y2.shape)
print(y3.shape)
print(y4.shape)