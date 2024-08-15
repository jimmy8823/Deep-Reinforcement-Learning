from stable_baselines3 import PPO
from torch import device, cuda, load
import numpy as np
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CM_VAE.CM_VAE_ver3 import CmVAE 
from env import ueEnv
import psutil
import time

class pipline_PPO():
    def __init__(self):
        self.device = device('cuda:0' if cuda.is_available() else 'cpu')
        print(self.device)
        self.cmvae = CmVAE(input_dim=2,semantic_dim=1,validation=True)
        self.cmvae.load_state_dict(load("D:/CODE/Python/AirSim/CM_VAE/mix_origin_depth/cmvae_50.pth"))
        self.cmvae.eval()
        self.agent = PPO.load("D:\\CODE\\Python\\AirSim\\PPO\\PPO_CMVAE\\model\\quad_land_2_100000_steps",device=self.device)
        
    def inference(self, images, info):        
        latent = self.cmvae(images)
        latent = latent.cpu().detach().numpy()
        cat_input = np.concatenate((latent,info),axis=1)
        action, _states = self.agent.predict(cat_input)
        return action
    
    
def test_ppo():
    """env = ueEnv()
    agent = pipline_PPO()
    done = False
    step = 0
    obs,info = env.reset_test()
    while not done:
        action = agent.inference(obs,info)
        nx_obs, nx_info, reward, done, collision, exceed, success = env.step(action,step)
        step += 1
        obs = nx_obs
        info = nx_info
    """
    print('Torch RAM memory used:', psutil.virtual_memory()[2])
    print('Torch RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    print('-----------------------------------------------------------------')

    for i in range(5):
        agent = pipline_PPO()
        print('Torch RAM memory used:', psutil.virtual_memory()[2])
        print('Torch RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
        images = np.random.random((1,2,144,256)).astype(np.float32)
        info = np.random.random((1,7)).astype(np.float32)
        start = time.time()
        action = agent.inference(images,info)
        end = time.time()
        print("[Torch] PPO+CMVAE inference take times : {}".format(end-start))

test_ppo()