import torch as T
from vae import VAE
import numpy as np
import re
import cv2
import PIL.Image


device = T.device('cuda' if T.cuda.is_available() else 'cpu')
def read_pfm(file):
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
    
def test():
    idx = 1
    file_path = "D:/CODE/Python/AirSim/CM_VAE/training_data/depth/depth_image_" + str(idx) + ".pfm"
    vae = VAE(1,8192,64)
    vae.load_checkpoint("vae_100.pth")
    depth_image, scale = read_pfm(file_path)
    depth_image = np.ascontiguousarray(depth_image)
    x = T.from_numpy(depth_image)
    x = x.unsqueeze(0)
    x = x.unsqueeze(0).to(device)
    x[x>25] = 25
    x /= 25 
    print(x.shape)
    recon, z, mu, logvar = vae(x)
    recon = recon.squeeze(0)
    recon = recon.squeeze(0).cpu().detach()
    print(recon)
    recon *= 25
    recon = np.array(recon, dtype=np.float32)

    depth_image = np.interp(depth_image, (0, 25), (0, 255))
    recon_image = np.interp(recon, (0, 25), (0, 255))
    cv2.imshow("input_depth", depth_image.astype('uint8'))
    cv2.imshow("recon_depht", recon_image.astype('uint8'))
    cv2.waitKey(0)
test()