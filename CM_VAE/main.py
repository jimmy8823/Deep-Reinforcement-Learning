import torch
import torch.nn.functional as F
import torch.nn as nn
from CM_VAE_ver2 import CmVAE
import numpy as np
from torchvision import transforms
import PIL.Image
import pandas as pd
import os
from torch.utils.data import Dataset,DataLoader
import cv2
import re
from torchsummary import summary
import skimage as ski

class CMVAE_Dataset(Dataset):
     # dataset loading
    def __init__(self,root_dir, data_dir, transform=True, depth_transfrom=True,seg_transform=True):
        self.root_dir = root_dir
        self.data_dir = pd.read_csv(os.path.join(root_dir, data_dir), names=['gray', 'depth', 'seg'])
        self.transform = transform
        self.depth_transform = depth_transfrom
        self.seg_transform = seg_transform
        # noisy
        self.mean = 0
        self.std_dev = 25
        # blur
        self.sigma = 1
        self.random_seed = 0
        """
        # print(len(self.data_dir))
        for i in range(len(self.data_dir)):
            img_path = os.path.join(self.root_dir, self.data_dir['image'][i])
            image = cv2.imread(img_path)
            self.images.append(image)
            label_path = os.path.join(self.root_dir, self.data_dir['label'][i])
            label = cv2.imread(label_path)
            self.labels.append(label)
        """
    # working for indexing
        
    def __getitem__(self, idx):
        gray_path = os.path.join(self.root_dir, self.data_dir['gray'][idx])
        # image = read_image(img_path)
        gray_image = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        gray_image = np.expand_dims(gray_image,axis=0)
        #pfm format
        depth_path = os.path.join(self.root_dir, self.data_dir['depth'][idx])
        # label = read_image(label_path)
        depth_image , scale = self.read_pfm(depth_path)

        seg_path = os.path.join(self.root_dir, self.data_dir['seg'][idx])
        # label = read_image(label_path)
        seg_image = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

        if self.random_seed % 5 :
            gray_image = self.blur_image(gray_image)

        if self.random_seed % 7 :
            gray_image = self.add_gaussian_noise(gray_image)

        if self.transform is True:
            transform = transforms.Compose([
                #transforms.ToPILImage(),
                #transforms.Resize([144,256]),
                transforms.ToTensor()
            ])
            #gray_image = transform(gray_image)

        if self.depth_transform is True:
            #depth_image = np.reshape(depth_image ,(1,144,256))
            depth_image = np.ascontiguousarray(depth_image)
            depth_image = torch.from_numpy(depth_image)
            depth_image = np.expand_dims(depth_image,axis=0)
            #depth_image = torch.tensor(depth_image,dtype = torch.float32)
            #depth_image = self.blur_image(depth_image)
            depth_image[depth_image>25] = 25

        #noramlize
        depth_image/=25 
        gray_image = gray_image/255
        seg_image = transform(seg_image)
        
        self.random_seed += 1
        #print(gray_path)
        return gray_image, depth_image, seg_image

    # return the length of our dataset
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

    def add_gaussian_noise(self, image): #針對影像加入高斯雜訊
        c, row, col = image.shape
        gauss = np.random.normal(self.mean, self.std_dev, (row, col))
        noisy = image + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def blur_image(self, image): #針對影像加入模糊效果
        blurred = ski.filters.gaussian(
        image, sigma=(self.sigma, self.sigma), truncate=3.5, channel_axis=-1)
        return blurred
    
def average(lst): 
    return  sum(lst) / len(lst) 

def count_params(model):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits
        print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

def main():
    # Hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gray_dim = 1  # grayscale dimensions
    depth_dim = 1  # depth dimensions
    semantic_dim = 1 # binary semantic pixel value 0 or 255
    hidden_dim = 5120
    latent_dim = 128
    learning_rate = 0.001
    epochs = 50
    batch_size = 100
    root_dir = "D:/CODE/Python/AirSim/CM_VAE/"
    train_dir = 'training_data/CM_VAE_training.csv'
    validation_dir = 'training_data/CM_VAE_validation.csv'
    train_dataset = CMVAE_Dataset(root_dir, train_dir, transform=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CMVAE_Dataset(root_dir, validation_dir, transform=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Create CM-VAE model
    cmvae = CmVAE(input_dim=2, semantic_dim=semantic_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(cmvae.parameters(), lr=learning_rate)  
    #torch.optim.SGD(cmvae.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=10, verbose=True)
    count_params(cmvae)

    summary(cmvae, (2, 144, 256))

    # Training loop 
        
    loss_history = []
    for epoch in range(epochs):
        training_loss = []
        val_loss = []
        max_recon_loss = 0
        #with autograd.detect_anomaly():
        for idx, (gray, depth, seg) in enumerate(train_loader):
            seg = seg.to(device)#.view(batch_size,-1)
            #print(seg.shape)
            # feed to CMVAE
            concat_x = np.concatenate((gray,depth),axis=1) # 1, 2 ,144 ,256
            reconstruction, kl = cmvae(concat_x)
            reconstruction_loss = F.binary_cross_entropy_with_logits(reconstruction, seg,reduction="sum")
            loss = reconstruction_loss + kl
            
            max_recon_loss = max(reconstruction_loss,max_recon_loss)
            #print(reconstruction.shape)
            #print(seg.shape)
            # print(f" loss:,{loss.item():0.4f},reconstruction_loss{reconstruction_loss : 0.4f}, kl :{kl:0.4f}")
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            
            #print(loss)
        with torch.no_grad(): # validation
            for idx, (gray, depth, seg) in enumerate(val_loader):
                seg = seg.to(device)
                concat_x = np.concatenate((gray,depth),axis=1) # 1, 2 ,144 ,256
                reconstruction, kl= cmvae(concat_x)
                reconstruction_loss = F.binary_cross_entropy_with_logits(reconstruction, seg,reduction="sum")
                loss = reconstruction_loss + kl
                #print(f" val_loss:,{loss.item():0.4f},reconstruction_loss{reconstruction_loss : 0.4f}, gray_kl :{gray_kl: 0.4f} , depth_kl : {depth_kl:0.4f}")
                val_loss.append(loss.item())
            scheduler.step(average(val_loss))
        if (epoch+1) % 10 == 0:
            cmvae.save_checkpoint(f"D:/CODE/Python/AirSim/CM_VAE/cmvae_{epoch+1}.pth")
        loss_history.append(f"{average(training_loss): 0.4f}")
        print(f'Epoch: {epoch+1}, loss: {average(training_loss): 0.4f}, reconstruct loss: {max_recon_loss: 0.4f}, val loss: {average(val_loss): 0.4f}')

    f = open("result.txt", "w")
    for idx , loss in enumerate(loss_history,1):
       f.write(f"{idx}, {loss}\n")
    f.close()
    
if __name__ == "__main__":
    main()

