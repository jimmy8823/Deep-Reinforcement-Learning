import time
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter 
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import datetime
from ppo import PPO
from env import ueEnv
from CM_VAE.CM_VAE_ver2 import CmVAE
from env import ueEnv

class Agent():
    ################ PPO hyperparameters ################
    def __init__(self) :
        self.max_episodes = 2000
        self.save_interval = 50
        self.episode = -1
        #update_freq = 4      # update policy every n timesteps
        self.K_epochs = 300               # update policy for K epochs
        self.eps_clip = 0.2              # clip parameter for PPO
        self.gamma = 0.99                # discount factor
        self.latent_dim = 128
        self.uav_info_dim = 7
        self.acion_dim = 7
        self.max_steps = 800
        self.collision_count = 0
        self.t_exceed_count = 0
        self.success_count = 0
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.cmvae_path = "D:/CODE/Python/AirSim/CM_VAE/mix_origin_depth/cmvae_50.pth"
        self.checkpoint = "2024_04_18-02_19_14" # test or continue training
        self.continues = True

        self.lr_actor = 0.0003       # learning rate for actor network
        self.lr_critic = 0.001       # learning rate for critic network

        self.env = ueEnv()
        self.path = (datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_path = "/model"

        print("[+] Loading VAE ...", flush=True)
        self.cmvae = CmVAE(input_dim=2,semantic_dim=1,validation=True)
        self.cmvae.load_state_dict(T.load(self.cmvae_path))
        self.cmvae.eval()

        self.ppo_agent = PPO(state_dim=self.latent_dim+self.uav_info_dim, action_dim=self.acion_dim, \
                         lr_actor=self.lr_actor, lr_critic=self.lr_critic, gamma=self.gamma, \
                         K_epochs=self.K_epochs, eps_clip=self.eps_clip, record_path=self.path)
        
        if T.cuda.is_available():
            print('[+] Using device:', T.cuda.get_device_name(0), flush=True)
        else:
            print("Using CPU")
            
################# training procedure ################

# initialize a PPO agent
    def train(self):
        print("[+] Start Training...", flush=True)

        if not self.continues:
            self.episode = 1
            os.mkdir(self.path)
            self.model_path = self.path + self.model_path
            os.mkdir(self.model_path)
            print("Training from scratch")
        else:
            self.path = self.checkpoint
            self.ppo_agent.record_path = self.checkpoint
            with open(self.path + '/result.txt') as f:
                last_line = f.readlines()
            if len(last_line) != 0:
                last_line = last_line[-1]
                self.episode = int(last_line.split(",")[0].split(":")[1]) +1
            self.ppo_agent.load_models()
        write_path = self.path + '/logs/PPO'
        self.writer = SummaryWriter(write_path)

        while self.episode <= self.max_episodes:
            start = time.time()
            obs, info = self.env.reset()
            obs = T.tensor(obs, dtype=T.float32, device=self.device)
            info = T.tensor(info, dtype=T.float32, device=self.device)
            latent = self.cmvae(obs) # [1, latent_dim]
            step = 0
            episode_reward = 0
            done = False
            success = False
            exceed = False
            episode_loss = 0
            terminal_state = ""
            while True:
                # select action with policy
                state = T.cat((latent,info),axis=1)
                action = self.ppo_agent.select_action(state)
                nx_obs, nx_info, reward, done, collision, exceed, success = self.env.step(action, step)
                nx_obs = T.tensor(nx_obs, dtype=T.float32, device=self.device)
                nx_info = T.tensor(nx_info, dtype=T.float32, device=self.device)
                nx_latent = self.cmvae(nx_obs) # [1, latent_dim]

                # saving reward and is_terminals
                self.ppo_agent.buffer.rewards.append(reward)
                self.ppo_agent.buffer.is_terminals.append(done)
                
                episode_reward += reward
                
                obs = nx_obs
                info = nx_info
                latent = nx_latent

                if step == self.max_steps: 
                    done = 1
                else:
                    step += 1

                if done :
                    """
                    for t in tqdm(range(0, step)):
                        loss = self.learn()
                        episode_loss += loss
                    """
                    if self.episode % 10 == 0:
                        episode_loss = self.ppo_agent.update()
                    
                    mean_rewards = round(episode_reward / step, 2)
                    #print(self.episode, episode_reward, mean_rewards, step, episode_loss)
                    print("-----------------------------------------------------", flush=True)
                    print("episode:{0}, reward: {1:.2f}, mean reward: {2}, total steps: {3}, loss: {4:.4f}".format(self.episode, episode_reward, mean_rewards, step, episode_loss), flush=True)

                    if collision:
                        self.collision_count += 1
                        terminal_state = "Collision"
                        print("Result : Collision")
                    elif exceed:
                        self.t_exceed_count += 1
                        terminal_state = "Time exceed"
                        print("Result : Time exceed")
                    elif success:
                        self.success_count += 1
                        terminal_state = "Success landing"
                        print("Result : Success landing")
                    
                    txt = self.path + "/result.txt"
                    with open(txt, 'a') as file:
                        file.write("episode:{0}, reward: {1:.4f}, mean reward: {2}, total steps: {3}, Result: {4}\n".format(self.episode, episode_reward, mean_rewards, step, terminal_state))
                    
                    self.writer.add_scalar('reward', episode_reward, self.episode)
                    self.writer.add_scalar('mean reward', mean_rewards, self.episode)
                    self.writer.add_scalar('total steps', step, self.episode)
                    self.writer.add_scalar('avg_loss', episode_loss, self.episode)

                    if self.episode % self.save_interval == 0:
                        self.ppo_agent.save_models(self.episode)

                    if self.episode % 50 == 0:

                        self.writer.add_scalars('terminal_state',{
                            'Sucess' : self.success_count,
                            'Collision' : self.collision_count,
                            'Time exceed' : self.t_exceed_count
                        },self.episode/50)
                        self.success_count = 0
                        self.collision_count = 0
                        self.t_exceed_count = 0
                    self.episode += 1

                    end = time.time()
                    stopWatch = end - start
                    print("Episode is done, episode time: ", stopWatch)

                    break

ppo = Agent()
ppo.train()