import torch.nn.functional as F
import torch
import torch.nn as nn
from CM_VAE_ver2 import CmVAE
from VAE import VAE
import cv2
import numpy as np
import airsim
import re
from torchvision import transforms
from torchsummary import summary
import skimage as ski

test_idx = 23812
vae_path = "D:/CODE/Python/AirSim/CM_VAE/result/exp 1/vae.pth"
cmvae_path = "D:/CODE/Python/AirSim/CM_VAE/mix_noise/cmvae_40.pth"
mean = 0
std_dev = 25
sigma = 3

def read_pfm(file): #讀取深度影像pfm
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

def test_vae():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    depth_dir = f'./Data/training_data/depth/depth_image_{test_idx}.pfm'
    gray_dir = f'./Data/training_data/gray/gray_image_{test_idx}.png'
    seg_dir = f'./Data/training_data/seg/bin_seg_image_{test_idx}.png'
    gray_dim = 1  # grayscale dimensions
    depth_dim = 1  # depth dimensions
    semantic_dim = 1 # binary semantic pixel value 0 or 255
    hidden_dim = 6144  
    latent_dim = 64
    threshold = 0.5
    vae = VAE(1,semantic_dim,hidden_dim,latent_dim,validation=False).to(device)
    vae.load_checkpoint(vae_path)

    gray_image = cv2.imread(gray_dir, cv2.IMREAD_GRAYSCALE)
    gray_image = np.expand_dims(gray_image,axis=0)
    gray_image = np.expand_dims(gray_image,axis=0)
    gray_image = torch.tensor(gray_image,dtype=torch.float32).to(device)
       
    gray_image/=255

    print(gray_image)

    depth_image , scale = read_pfm(depth_dir)
    depth_image = np.ascontiguousarray(depth_image)
    depth_image = torch.from_numpy(depth_image)
    depth_image = np.expand_dims(depth_image,axis=0)
    depth_image = np.expand_dims(depth_image,axis=0)
    depth_image = torch.tensor(depth_image,dtype = torch.float32).to(device)
    depth_image[depth_image>25] = 25
    # depth_image/=25 #noramlize
    print(depth_image)

    seg_image = cv2.imread(seg_dir, cv2.IMREAD_GRAYSCALE)
    seg_image = np.expand_dims(seg_image,axis=0)
    seg_image = np.expand_dims(seg_image,axis=0)
    seg_image = torch.tensor(seg_image,dtype=torch.float32).to(device)
    #seg_image/=255
    print(seg_image)
    #x = torch.cat((gray_image,depth_image),dim=1)
    reconstruction , _ , _ = vae(gray_image)
    
    print(reconstruction)
    print(torch.min(reconstruction))
    print(torch.max(reconstruction))
    """
    reconstructiown[reconstruction>=threshold]=255
    reconstruction[reconstruction<threshold]=0
    
    """
    reconstruction *= 255
    reconstruction = reconstruction.detach().squeeze().cpu().numpy()

    #print(reconstruction.shape)
    airsim.write_png(f"reconstruction_vae_{test_idx}.png" ,reconstruction)

def test_cmvae():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    depth_dir = f'D:/CODE/Python/AirSim/CM_VAE/training_data/depth/depth_image_{test_idx}.pfm'
    gray_dir = f'D:/CODE/Python/AirSim/CM_VAE/training_data/gray/gray_image_{test_idx}.png'
    seg_dir = f'D:/CODE/Python/AirSim/CM_VAE/training_data/seg/bin_seg_image_{test_idx}.png'
    gray_dim = 1  # grayscale dimensions
    depth_dim = 1  # depth dimensions
    semantic_dim = 1 # binary semantic pixel value 0 or 255
    hidden_dim = 5120  
    latent_dim = 128
    threshold = 0.5
    #cmvae = CmVAE(2,semantic_dim,hidden_dim,latent_dim,validation=False).to(device)
    cmvae = CmVAE(input_dim=2, semantic_dim=semantic_dim,hidden_dim=hidden_dim,latent_dim=latent_dim)
    cmvae.load_checkpoint(cmvae_path)
    summary(cmvae, (2, 144, 256))
    
    gray_image = cv2.imread(gray_dir, cv2.IMREAD_GRAYSCALE)
    gray_image = add_gaussian_noise(gray_image, mean, std_dev)
    gray_image = np.expand_dims(gray_image,axis=0)
    gray_image = np.expand_dims(gray_image,axis=0)
    gray_image = torch.tensor(gray_image,dtype=torch.float32).to(device)
    gray_image/=255

    #print(gray_image)

    depth_image , scale = read_pfm(depth_dir)
    depth_image = np.ascontiguousarray(depth_image)
    depth_image = torch.from_numpy(depth_image)
    depth_image = np.expand_dims(depth_image,axis=0)
    depth_image = np.expand_dims(depth_image,axis=0)
    depth_image = torch.tensor(depth_image,dtype = torch.float32).to(device)
    depth_image[depth_image>25] = 25
    depth_image/=25 #noramlize

    seg_image = cv2.imread(seg_dir, cv2.IMREAD_GRAYSCALE)
    seg_image = np.expand_dims(seg_image,axis=0)
    seg_image = np.expand_dims(seg_image,axis=0)
    seg_image = torch.tensor(seg_image,dtype=torch.float32).to(device)
    seg_image/=255
    #print(seg_image)

    x = torch.cat((gray_image,depth_image),dim=1)
    reconstruction , _ = cmvae(x)

    reconstruction[reconstruction>=threshold]=255
    reconstruction[reconstruction<threshold]=0

    reconstruction = reconstruction.detach().squeeze().cpu().numpy()

    #print(reconstruction.shape)
    #airsim.write_png(f"D:/CODE/Python/AirSim/CM_VAE/reconstruction_cmvae_{test_idx}.png",reconstruction)

def pfm2png():
    depth_dir = f'D:/CODE/Python/AirSim/CM_VAE/training_data/depth/depth_image_{test_idx}.pfm'
    depth_image , scale = read_pfm(depth_dir)
    print(depth_image)
    u16 = (65535*(depth_image - np.min(depth_image))/np.ptp(depth_image)).astype(np.uint16)  
    airsim.write_png(f'D:/CODE/Python/AirSim/CM_VAE/depth_{test_idx}.png', u16)

def add_gaussian_noise(image, mean, std_dev):
    row, col = image.shape
    gauss = np.random.normal(mean, std_dev, (row, col))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def blur_image(image, sigma):
    blurred = ski.filters.gaussian(
    image, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
    return blurred
#test_vae()
#test_cmvae()
pfm2png()
test_cmvae()
