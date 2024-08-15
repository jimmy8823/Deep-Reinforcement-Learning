import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class VAE(nn.Module):
    def __init__(self, input_dim, semantic_dim, hidden_dim=6144, latent_dim=128, validation=False):
        super(VAE, self).__init__()
        
        self.validation = validation

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            Flatten()
        )

        self.decoder = nn.Sequential(   
            nn.Unflatten(1, (512, 3, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=[2, 4], stride=2, padding=1),
            # 4*8
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=[4, 4], stride=2, padding=1),
            # 8*16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=[4, 4], stride=2, padding=1),
            # 16*32
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=[4, 4], stride=2, padding=1),
            # 32*64
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=[8, 4], stride=2, padding=1),
            # 68*128
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, input_dim, kernel_size=[12, 4], stride=2, padding=1),
            nn.Sigmoid()
        )
        
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_d = nn.Linear(latent_dim, hidden_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                #m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                m.weight.data = nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def bottleneck(self, h):
        mu, logvar = self.fc_mean(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc_d(z)
        z = self.decoder(z)
        return z
    
    def forward(self, gray):
        z, mu, logvar = self.encode(gray)
        z = self.decode(z)
        return z, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x,reduction='sum')
        KLD =  torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(),dim=1),dim=0)
        return BCE + KLD, BCE, KLD

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

class AE(nn.Module):
    def __init__(self, input_dim, semantic_dim, hidden_dim=6144, latent_dim=64, validation=False):
        super(AE, self).__init__()
        
        self.validation = validation

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            Flatten()
        )

        self.decoder = nn.Sequential(   
            nn.Unflatten(1, (512, 3, 4)),

            nn.ConvTranspose2d(512, 256, kernel_size=[2, 4], stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=[4, 4], stride=2, padding=1),
            # 8*16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=[4, 4], stride=2, padding=1),
            # 16*32
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=[4, 4], stride=2, padding=1),
            # 32*64
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=[8, 4], stride=2, padding=1),
            # 68*128
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, input_dim, kernel_size=[12, 4], stride=2, padding=1),
            #nn.Sigmoid()
        )
        
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.fc_d = nn.Linear(latent_dim, hidden_dim)
        self.to(device)

    def forward(self,x):
        z = self.fc(self.encoder(x))
        recons = self.decoder(self.fc_d(z))
        return recons
    
    def loss_function(self, recon_x,x):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x,reduction='sum')
        return BCE
    
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))