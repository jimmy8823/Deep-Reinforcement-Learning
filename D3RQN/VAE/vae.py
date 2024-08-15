import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VAE(nn.Module):
    def __init__(self, image_channel=1, h_dim=7680, z_dim=64):
        super(VAE, self).__init__()
        '''
        name = "vae"
        cwd = os.getcwd()
        self.checkpoint_file = os.path.join(cwd, name)
        if not os.path.exists(self.checkpoint_file):
            os.mkdir(name)
        '''
        self.encoder = nn.Sequential(
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
            #nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            #nn.LeakyReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        # ConvTranspose2d: o = s * (i - 1) - p * 2 + k
        
        self.decoder = nn.Sequential(   
            nn.Unflatten(1, (256, 4, 8)), # 256 4 8 
            #nn.ConvTranspose2d(512, 256, kernel_size=[5, 4], stride=2, padding=1),
            #nn.LeakyReLU(), 
            nn.ConvTranspose2d(256, 128, kernel_size=[4, 4], stride=2, padding=1),
            nn.LeakyReLU(), # 4 4
            nn.ConvTranspose2d(128, 64, kernel_size=[6, 4], stride=2, padding=1),
            nn.LeakyReLU(), # 6 4
            nn.ConvTranspose2d(64, 32, kernel_size=[4, 4], stride=2, padding=1),
            nn.LeakyReLU(), # 4 4
            nn.ConvTranspose2d(32, 16, kernel_size=[4, 4], stride=2, padding=1),
            nn.LeakyReLU(), # 4 4
            nn.ConvTranspose2d(16, image_channel, kernel_size=[4, 4], stride=2, padding=1),
            nn.Sigmoid() # 4 4
        )
        
        self.to(device)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                m.weight.data = nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        
    def reparameterize(self, mu, logvar):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(device)
        return mu + std * esp
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return self.decode(z), z, mu, logvar
    
    def val(self, x):
        z, _, _ = self.encode(x)
        return z
        
    def loss_fn(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - (torch.clamp(logvar, max=99999)).exp()))

        return BCE + KLD, BCE, KLD
    
    def save_checkpoint(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_checkpoint(self, file_path=""):
        if file_path == "":
            files = glob.glob(self.checkpoint_file + '\\*.pt')
            files.sort(key=os.path.getmtime)
            self.load_state_dict(torch.load(files[-1]))
    
        else:
            self.load_state_dict(torch.load(file_path))