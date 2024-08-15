import numpy as np
import cv2
import skimage as ski
import re
import PIL.Image

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

def add_gaussian_noise(image, mean, std_dev):
    row, col = image.shape
    gauss = np.random.normal(mean, std_dev, (row, col))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def blur_image(image, sigma):
    blurred = ski.filters.gaussian(
    image, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
    return blurred

test_idx = 18000
depth_dir = f'D:/CODE/Python/AirSim/CM_VAE/training_data/depth/depth_image_{test_idx}.pfm'
gray_dir = f'D:/CODE/Python/AirSim/CM_VAE/training_data/gray/gray_image_{test_idx}.png'
seg_dir = f'D:/CODE/Python/AirSim/CM_VAE/training_data/seg/bin_seg_image_{test_idx}.png'
# noisy
mean = 0
std_dev = 25
# blur
sigma = 1

gray_image = cv2.imread(gray_dir, cv2.IMREAD_GRAYSCALE)
depth_image, scale = read_pfm(depth_dir)
depth_image = np.ascontiguousarray(depth_image)


print(depth_image)

noisy_image = add_gaussian_noise(gray_image, mean, std_dev)
blurred_gray = blur_image(gray_image,sigma)*255
mix_gray = blur_image(noisy_image,sigma)*255

blurred_depth = blur_image(depth_image,sigma)

depth_image[depth_image>25] = 25
depth_image = np.array(PIL.Image.fromarray(depth_image).convert("L"))
depth_image = depth_image/25*255 
blurred_depth[blurred_depth>25] = 25
blurred_depth = np.array(PIL.Image.fromarray(blurred_depth).convert("L"))

cv2.imwrite("Original Image.png",gray_image)
cv2.imwrite("Noisy Image.png",noisy_image)
cv2.imwrite("blur Image.png",blurred_gray)
cv2.imwrite("mix Image.png", mix_gray)
cv2.imwrite("depth.png",depth_image)

cv2.imshow('Original Image', gray_image)
cv2.imshow('Noisy Image', noisy_image)
cv2.imshow("blur Image", blurred_gray)
cv2.imshow("Original Depth",depth_image)
cv2.imshow("blur Depth",blurred_depth)
cv2.imshow("mix Image", mix_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()