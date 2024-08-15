import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import torchvision.datasets as datasets
import numpy as np
from torchvision import transforms
import PIL.Image
import pandas as pd
import os
from torch.utils.data import Dataset,DataLoader

from vae import VAE
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def criterion(predict, target, ave, log_dev):
  bce_loss = F.binary_cross_entropy(predict, target, reduction='sum')
  kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
  loss = bce_loss + kl_loss
  return loss

def main():
    # Hyperparameters
    
    input_dim = 784
    latent_dim = 2
    learning_rate = 0.001
    epochs = 20
    BATCH_SIZE = 100

    trainval_data = datasets.MNIST("./data", 
                    train=True, 
                    download=True, 
                    transform=transforms.ToTensor())

    train_size = int(len(trainval_data) * 0.8)
    val_size = int(len(trainval_data) * 0.2)
    train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])

    train_loader = DataLoader(dataset=train_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=0)

    val_loader = DataLoader(dataset=val_data,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            num_workers=0)

    print("train data size: ",len(train_data))   #train data size:  48000
    print("train iteration number: ",len(train_data)//BATCH_SIZE)   #train iteration number:  480
    print("val data size: ",len(val_data))   #val data size:  12000
    print("val iteration number: ",len(val_data)//BATCH_SIZE)   #val iteration number:  120
    model = VAE(latent_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    
    #torch.optim.SGD(cmvae.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)

    history = {"train_loss": [], "val_loss": [], "ave": [], "log_dev": [], "z": [], "labels":[]}

    # Training loop 
    for epoch in range(epochs):
        model.train()
        for i, (x, labels) in enumerate(train_loader):
            input = x.to(device).view(-1, 28*28).to(torch.float32)
            output, z, ave, log_dev = model(input)

            history["ave"].append(ave)
            history["log_dev"].append(log_dev)
            history["z"].append(z)
            history["labels"].append(labels)
            loss = criterion(output, input, ave, log_dev)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 50 == 0:
                print(f'Epoch: {epoch+1}, loss: {loss: 0.4f}')
            history["train_loss"].append(loss)

        model.eval()
        with torch.no_grad():
            for i, (x, labels) in enumerate(val_loader):
                input = x.to(device).view(-1, 28*28).to(torch.float32)
                output, z, ave, log_dev = model(input)

                loss = criterion(output, input, ave, log_dev)
                history["val_loss"].append(loss)
            
            print(f'Epoch: {epoch+1}, val_loss: {loss: 0.4f}')
        
        scheduler.step()
    ave_tensor = torch.stack(history["ave"])
    log_var_tensor = torch.stack(history["log_dev"])
    z_tensor = torch.stack(history["z"])
    labels_tensor = torch.stack(history["labels"])
    print(ave_tensor.size())   #torch.Size([9600, 100, 2])
    print(log_var_tensor.size())   #torch.Size([9600, 100, 2])
    print(z_tensor.size())   #torch.Size([9600, 100, 2])
    print(labels_tensor.size())   #torch.Size([9600, 100])

    ave_np = ave_tensor.to('cpu').detach().numpy().copy()
    log_var_np = log_var_tensor.to('cpu').detach().numpy().copy()
    z_np = z_tensor.to('cpu').detach().numpy().copy()
    labels_np = labels_tensor.to('cpu').detach().numpy().copy()
    print(ave_np.shape)   #(9600, 100, 2)
    print(log_var_np.shape)   #(9600, 100, 2)
    print(z_np.shape)   #(9600, 100, 2)
    print(labels_np.shape)   #(9600, 100)
    model.to("cpu")
    batch_num = 9580
    label = 0
    x_zero_mean = np.mean(ave_np[batch_num:,:,0][labels_np[batch_num:,:] == label])   #x軸の平均値
    y_zero_mean = np.mean(ave_np[batch_num:,:,1][labels_np[batch_num:,:] == label])   #y軸の平均値
    z_zero = torch.tensor([x_zero_mean,y_zero_mean], dtype = torch.float32)

    output = model.decoder(z_zero)
    np_output = output.to('cpu').detach().numpy().copy()
    np_image = np.reshape(np_output, (28, 28))
    plt.imshow(np_image, cmap='gray')


if __name__ == "__main__":
    main()

