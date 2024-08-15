import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            Flatten(),
            nn.ReLU()
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
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fcmean = nn.Linear(128, latent_dim)
        self.fclogvar = nn.Linear(128, latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                #m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                m.weight.data = nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

    def forward(self, x):
        h = self.cnn(x) #
        o = self.fc1(h) # 128
        mean = self.fcmean(o)
        logvar = self.fclogvar(o)
        #print(concat)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, output_dim=1, hidden_dim=5120, latent_dim=128):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim) # 8192
        self.decode = nn.Sequential(   
            nn.Unflatten(1, (128, 5, 8)),
            #nn.ConvTranspose2d(512, 256, kernel_size=[5, 4], stride=2, padding=1),
            #nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=[4, 4], stride=2, padding=1),
            # 10*16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=[2, 4], stride=2, padding=1),
            # 18*32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=[4, 4], stride=2, padding=1),
            # 36*64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=[4, 4], stride=2, padding=1),
            # 72*128
            nn.ReLU(),
            nn.ConvTranspose2d(16, output_dim, kernel_size=[4, 4], stride=2, padding=1),
            # 144*256
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                #m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                m.weight.data = nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

    def forward(self, z):
        x = torch.relu(self.fc(z)) # not sure if relu is needed
        reconstruction = self.decode(x)
        return reconstruction

class CmVAE(nn.Module):
    def __init__(self, input_dim, semantic_dim, hidden_dim=5120, latent_dim=128, validation=False):
        super(CmVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim,latent_dim)
        self.decoder = Decoder(semantic_dim , hidden_dim, latent_dim)
        self.validation = validation
        self.to(device)


    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        mean, logvar = self.encoder(x)
        #print(mean,logvar)
        z = self.sample_latent(mean, logvar) 
        
        if self.validation:
            return z
        
        reconstruction = self.decoder(z)
        kl = self.kl_divergence(mean, logvar)
        return reconstruction, kl

    def sample_latent(self, mean, logvar):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mean.size()).to(device)
        return mean + eps * std
    
    def kl_divergence(self, mean, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(),dim=1),dim=0)

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)
        print("save checkpoint :")

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels ,kernel_size = 3, stride = 1, downsample = None):
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
    
class Conv2dSame(torch.nn.Conv2d):

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