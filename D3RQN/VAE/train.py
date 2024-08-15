from vae import VAE
import numpy as np
import torch as T
import pandas as pd
import os
from torch.utils.data import Dataset,DataLoader
import re
from tqdm import tqdm
import cv2

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

class VAE_Dataset(Dataset):
    def __init__(self,root_dir, data_dir):
        self.root_dir = root_dir
        self.data_dir = pd.read_csv(os.path.join(root_dir, data_dir), names=['gray', 'depth', 'seg'])

    def __getitem__(self, idx):
        depth_path = os.path.join(self.root_dir, self.data_dir['depth'][idx])
        depth_image , scale = self.read_pfm(depth_path)
        depth_image = np.ascontiguousarray(depth_image)
        depth_image = T.from_numpy(depth_image)
        depth_image = np.expand_dims(depth_image,axis=0)

        #noramlize
        depth_image[depth_image>25] = 25
        depth_image/=25 

        return depth_image
    
    def __len__(self):
        return len(self.data_dir)

    def read_pfm(self,file):
        file = open(file, 'rb')
        
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode('ascii') == 'PF':
            color = True    
        elif header.decode('ascii') == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.search(r'(\d+)\s(\d+)', file.readline().decode('ascii'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        return np.reshape(data, shape), scale
    
def average(lst): 
    return  sum(lst) / len(lst) 

def main():
    # hyperparameter 
    hidden_dim = 8192
    latent_dim = 64
    epochs = 100
    batch_size = 100
    learning_rate = 0.001

    # image_data path
    root_dir = "D:/CODE/Python/AirSim/CM_VAE/"
    train_dir = 'training_data/CM_VAE_training.csv'
    validation_dir = 'training_data/CM_VAE_validation.csv'
    train_dataset = VAE_Dataset(root_dir, train_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = VAE_Dataset(root_dir, validation_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # VAE Model 
    vae = VAE(1,hidden_dim,latent_dim)
    optimzer = T.optim.Adam(vae.parameters(), lr=learning_rate) 

    # Training loop
    for epoch in tqdm(range(epochs)):
        training_loss = []
        val_loss = []
        for idx, (depth_image) in tqdm(enumerate(train_loader)):
            depth_image = depth_image.to(device)
            recon, z, mean, logvar = vae(depth_image)
            loss, BCE, KLD = vae.loss_fn(recon, depth_image, mean, logvar)
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            training_loss.append(loss.item())
            
        with T.no_grad():
            for idx, (depth_image) in tqdm(enumerate(val_loader)):
                depth_image = depth_image.to(device)
                recon, z, mean, logvar = vae(depth_image)
                loss, BCE, KLD = vae.loss_fn(recon, depth_image, mean, logvar)
                val_loss.append(loss.item())

        if (epoch+1) % 10 == 0:
            vae.save_checkpoint(f"D:/CODE/Python/AirSim/D3RQN/VAE/vae_{epoch+1}.pth")
        print(f'Epoch: {epoch+1}, loss: {average(training_loss): 0.4f}, val loss: {average(training_loss): 0.4f}')
        
main()      