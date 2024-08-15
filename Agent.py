import os
import time
import cv2
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter 
from tqdm import tqdm
import datetime
from env import ueEnv
from D3QN import D3QN,Prioritized_replay,NStepMemory,D3QN_cnn
from CM_VAE.CM_VAE_ver2 import CmVAE
import random

class Agent():
    def __init__(self):
        self.max_episodes = 2000
        self.save_interval = 50
        self.episode = -1
        self.max_steps = 300
        self.batch_size = 32
        self.nstep = 4
        self.lr = 0.0001
        self.gamma = 0.99
        self.memory_size = 10000
        self.latent_dim = 128
        self.uav_info_dim = 7
        self.action_dim = 7
        self.collision_count = 0
        self.t_exceed_count = 0
        self.success_count = 0
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.cmvae_path = "D:/CODE/Python/AirSim/CM_VAE/mix_origin_depth/cmvae_50.pth"
        self.checkpoint = "2024_07_29-17_29_01" # test or continue training
        self.continues = False #if True continues from checkpoint dir

        # create airsim client
        self.env = ueEnv()
        self.path = (datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_path = "/model"
        # load cmvae and D3QN
        print("[+] Loading CMVAE ...", flush=True)
        self.cmvae = CmVAE(input_dim=2,semantic_dim=1,validation=True)
        self.cmvae.load_state_dict(T.load(self.cmvae_path))
        self.cmvae.eval()
        self.D3QN = D3QN(input_dim=(self.latent_dim+self.uav_info_dim), action_dim=self.action_dim,
                         lr=self.lr, gamma=self.gamma, epsilon=1.0, update_freq=5000, steps=0,
                         path=self.path)
        
        # init replay buffer
        self.memory = Prioritized_replay(obs_dim=self.latent_dim, info_dim=self.uav_info_dim, 
                                         memory_size=self.memory_size, batch_size=self.batch_size,
                                         cm_vae=True)
        self.nsmemory = NStepMemory(memory_size=self.max_steps, n_steps=self.nstep, gamma=self.gamma)
        if T.cuda.is_available():
            print('[+] Using device:', T.cuda.get_device_name(0), flush=True)
        else:
            print("Using CPU")
    
    def learn(self):
        # check if buffer transition enough to train
        if not self.memory.warming_up():
            return 0
        
        # sample batch of transition to learn
        b_idx, b_latent, b_info, b_action, b_rewards, b_nx_latent, b_nx_info, b_dones, ISWeight = self.memory.sample_buffer()
        b_latent = T.tensor(np.array(b_latent, dtype=np.float32), dtype=T.float, device=self.device)
        b_info = T.tensor(np.array(b_info, dtype=np.float32), dtype=T.float, device=self.device)
        b_action = T.tensor(np.array(b_action, dtype=np.float32), dtype=T.int64, device=self.device).unsqueeze(1)
        b_rewards = T.tensor(np.array(b_rewards, dtype=np.float32), dtype=T.float, device=self.device).unsqueeze(1)
        b_nx_latent = T.tensor(np.array(b_nx_latent, dtype=np.float32), dtype=T.float, device=self.device)
        b_nx_info = T.tensor(np.array(b_nx_info, dtype=np.float32), dtype=T.float, device=self.device)
        b_dones = T.tensor(np.array(b_dones, dtype=np.float32), dtype=T.float, device=self.device).unsqueeze(1)
        b_ISWeight = T.tensor(np.array(ISWeight, dtype=np.float32), dtype=T.float,  device=self.device).unsqueeze(1)

        with T.no_grad():
            nx_state = T.cat((b_nx_latent,b_nx_info),dim=1) # cat next state
            next_q_value = self.D3QN.target_q_net(nx_state) # target net 進行估計target value
            max_action = T.argmax(self.D3QN.q_net(nx_state),dim=1,keepdim=True) # double network (eval net 估計最大Q value作為選擇的action)
            q_targets = b_rewards + self.gamma * next_q_value.gather(1,max_action) * (1 - b_dones)
        state = T.cat((b_latent, b_info),dim=1)
        q_values = self.D3QN.q_net(state).gather(1,b_action)

        abs_errors = T.abs(q_values - q_targets).detach().cpu().numpy() #計算TD error

        self.memory.batch_update(b_idx, abs_errors) #更新priority

        loss = (b_ISWeight * F.mse_loss(q_values, q_targets.detach())).mean()
        self.D3QN.update(loss)

        return loss

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
            self.D3QN.record_path = self.checkpoint
            with open(self.path + '/result.txt') as f:
                last_line = f.readlines()
            if len(last_line) != 0:
                last_line = last_line[-1]
                self.episode = int(last_line.split(",")[0].split(":")[1]) +1
            self.D3QN.load_models()
        write_path = self.path + '/logs/D3QN'
        self.writer = SummaryWriter(write_path)
        
        while(self.episode <= self.max_episodes):
            start = time.time()
            obs, info = self.env.reset() # obs [1,2,144,256] ; info [1,7] numpy
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
                action = self.D3QN.choose_action(latent, info)
                nx_obs, nx_info, reward, done, collision, exceed, success = self.env.step(action, step)
                nx_obs = T.tensor(nx_obs, dtype=T.float32, device=self.device)
                nx_info = T.tensor(nx_info, dtype=T.float32, device=self.device)
                nx_latent = self.cmvae(nx_obs) # [1, latent_dim]

                self.nsmemory.push(latent, info, action, reward, nx_latent, nx_info, done)

                obs = nx_obs
                info = nx_info
                latent = nx_latent

                episode_reward += reward
                
                if step == self.max_steps: 
                    done = 1
                else:
                    step += 1

                if done :
                     # for s in range(steps - self.n_steps + 1):
                    for s in range(step):
                        latent, info, action, sum_reward, nx_latent, nx_infot, done = self.nsmemory.get(s)
                        self.memory.store_transition(latent, info, action, sum_reward, nx_latent, nx_infot, done)
                    self.nsmemory.reset()

                    for t in tqdm(range(0, step)):
                        loss = self.learn()
                        episode_loss += loss

                    episode_loss /= step
                    mean_rewards = round(episode_reward / step, 2)
                    print("-----------------------------------------------------", flush=True)
                    print("episode:{0}, reward: {1:.2f}, mean reward: {2}, total steps: {3}, loss: {4:.4f}, epsilon: {5:.2f}".format(self.episode, episode_reward, mean_rewards, step, episode_loss, self.D3QN.epsilon), flush=True)

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
                        file.write("episode:{0}, reward: {1:.4f}, mean reward: {2}, total steps: {3}, epsilon: {4:.3f}, Result: {5}\n".format(self.episode, episode_reward, mean_rewards, step, self.D3QN.epsilon, terminal_state))
                    
                    self.writer.add_scalar('reward', episode_reward, self.episode)
                    self.writer.add_scalar('mean reward', mean_rewards, self.episode)
                    self.writer.add_scalar('total steps', step, self.episode)
                    self.writer.add_scalar('avg_loss', episode_loss, self.episode)

                    if self.episode % self.save_interval == 0:
                        self.D3QN.save_models(self.episode)

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
                    print()

                    break
        
        print("total_done : {} step".format(self.D3QN.eps_step))

    def test(self):
        self.D3QN.validation = True
        self.D3QN.record_path = self.checkpoint
        self.D3QN.load_models()
        for i in range(158):
            obs, info = self.env.reset_test() # obs [1,2,144,256] ; info [1,7] numpy
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
            start = time.time()
            while True:
                action = self.D3QN.choose_action(latent, info)
                nx_obs, nx_info, reward, done, collision, exceed, success = self.env.step(action, step)
                nx_obs = T.tensor(nx_obs, dtype=T.float32, device=self.device)
                nx_info = T.tensor(nx_info, dtype=T.float32, device=self.device)
                nx_latent = self.cmvae(nx_obs) # [1, latent_dim]

                episode_reward += reward
                obs = nx_obs
                info = nx_info
                latent = nx_latent

                step += 1

                if done :
                    if collision:
                        self.collision_count += 1
                        terminal_state = "Collision"
                    elif exceed:
                        self.t_exceed_count += 1
                        terminal_state = "Time exceed"
                        
                    elif success:
                        self.success_count += 1
                        terminal_state = "Success landing"
                    
                    end = time.time()
                    stopWatch = end - start
                    print("-----------------------------------------------------", flush=True)
                    #print("episode:{0}, reward: {1:.2f}, total steps: {2}, loss: {3:.4f}, epsilon: {4:.2f}".format(self.episode, episode_reward, step, episode_loss, self.D3QN.epsilon), flush=True)
                    print("Episode is done, episode time: ", stopWatch)
                    print("Result : {}".format(terminal_state))

                    with open("test_result.txt","a")as f:
                        f.write("Episode time: {0:.2f}, Result: {1}\n".format(stopWatch,terminal_state))
                    break

class CNNAgent():
    def __init__(self):
        self.max_episodes = 4000
        self.save_interval = 50
        self.episode = -1
        self.max_steps = 300
        self.batch_size = 32
        self.nstep = 4
        self.lr = 0.0001
        self.gamma = 0.99
        self.memory_size = 5000
        self.uav_info_dim = 7
        self.action_dim = 7
        self.collision_count = 0
        self.t_exceed_count = 0
        self.success_count = 0
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #self.cmvae_path = "D:/CODE/Python/AirSim/CM_VAE/mix_origin_depth/cmvae_50.pth"
        self.checkpoint = "2024_08_04-21_04_29" # test or continue training
        self.continues = True #if True continues from checkpoint dir
        self.obs_dim = [1,2,144,256]

        # create airsim client
        self.env = ueEnv()
        self.path = (datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_path = "/model"

        self.D3QN = D3QN_cnn(self.obs_dim,self.uav_info_dim, action_dim=self.action_dim,
                         lr=self.lr, gamma=self.gamma, epsilon=1.0, update_freq=5000, steps=0,
                         path=self.path)
        
        # init replay buffer
        self.memory = Prioritized_replay(obs_dim=self.obs_dim, info_dim=self.uav_info_dim, 
                                         memory_size=self.memory_size, batch_size=self.batch_size,
                                         cm_vae=False)
        self.nsmemory = NStepMemory(memory_size=self.max_steps, n_steps=self.nstep, gamma=self.gamma)
        if T.cuda.is_available():
            print('[+] Using device:', T.cuda.get_device_name(0), flush=True)
        else:
            print("Using CPU")
    
    def learn(self):
        # check if buffer transition enough to train
        if not self.memory.warming_up():
            return 0
        
        # sample batch of transition to learn
        b_idx, b_obs, b_info, b_action, b_rewards, b_nx_obs, b_nx_info, b_dones, ISWeight = self.memory.sample_buffer()
        b_obs = T.tensor(np.array(b_obs, dtype=np.float32), dtype=T.float, device=self.device)
        b_info = T.tensor(np.array(b_info, dtype=np.float32), dtype=T.float, device=self.device)
        b_action = T.tensor(np.array(b_action, dtype=np.float32), dtype=T.int64, device=self.device).unsqueeze(1)
        b_rewards = T.tensor(np.array(b_rewards, dtype=np.float32), dtype=T.float, device=self.device).unsqueeze(1)
        b_nx_obs = T.tensor(np.array(b_nx_obs, dtype=np.float32), dtype=T.float, device=self.device)
        b_nx_info = T.tensor(np.array(b_nx_info, dtype=np.float32), dtype=T.float, device=self.device)
        b_dones = T.tensor(np.array(b_dones, dtype=np.float32), dtype=T.float, device=self.device).unsqueeze(1)
        b_ISWeight = T.tensor(np.array(ISWeight, dtype=np.float32), dtype=T.float,  device=self.device).unsqueeze(1)

        with T.no_grad():
            next_q_value = self.D3QN.target_q_net(b_nx_obs,b_nx_info) # target net 進行估計target value
            max_action = T.argmax(self.D3QN.q_net(b_nx_obs,b_nx_info),dim=1,keepdim=True) # double network (eval net 估計最大Q value作為選擇的action)
            q_targets = b_rewards + self.gamma * next_q_value.gather(1,max_action) * (1 - b_dones)
        q_values = self.D3QN.q_net(b_obs, b_info).gather(1,b_action)

        abs_errors = T.abs(q_values - q_targets).detach().cpu().numpy() #計算TD error

        self.memory.batch_update(b_idx, abs_errors) #更新priority

        loss = (b_ISWeight * F.mse_loss(q_values, q_targets.detach())).mean()
        self.D3QN.update(loss)

        return loss

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
            self.D3QN.record_path = self.checkpoint
            with open(self.path + '/result.txt') as f:
                last_line = f.readlines()
            if len(last_line) != 0:
                last_line = last_line[-1]
                self.episode = int(last_line.split(",")[0].split(":")[1]) +1
            self.D3QN.load_models()
        write_path = self.path + '/logs/D3QN'
        self.writer = SummaryWriter(write_path)
        
        while(self.episode <= self.max_episodes):
            start = time.time()
            obs, info = self.env.reset() # obs [1,2,144,256] ; info [1,7] numpy
            obs = T.tensor(obs, dtype=T.float32, device=self.device)
            info = T.tensor(info, dtype=T.float32, device=self.device)
            #latent = self.cmvae(obs) # [1, latent_dim]
            step = 0
            episode_reward = 0
            done = False
            success = False
            exceed = False
            episode_loss = 0
            terminal_state = ""
            while True:
                action = self.D3QN.choose_action(obs, info)
                nx_obs, nx_info, reward, done, collision, exceed, success = self.env.step(action, step)
                nx_obs = T.tensor(nx_obs, dtype=T.float32, device=self.device)
                nx_info = T.tensor(nx_info, dtype=T.float32, device=self.device)
               # nx_latent = self.cmvae(nx_obs) # [1, latent_dim]

                self.nsmemory.push(obs, info, action, reward, nx_obs, nx_info, done)

                obs = nx_obs
                info = nx_info

                episode_reward += reward
                
                if step == self.max_steps: 
                    done = 1
                else:
                    step += 1

                if done :
                     # for s in range(steps - self.n_steps + 1):
                    for s in range(step):
                        obs, info, action, sum_reward, nx_obs, nx_info, done = self.nsmemory.get(s)
                        self.memory.store_transition(obs, info, action, sum_reward, nx_obs, nx_info, done)
                    self.nsmemory.reset()

                    for t in tqdm(range(0, step)):
                        loss = self.learn()
                        episode_loss += loss

                    episode_loss /= step
                    mean_rewards = round(episode_reward / step, 2)
                    print("-----------------------------------------------------", flush=True)
                    print("episode:{0}, reward: {1:.2f}, mean reward: {2}, total steps: {3}, loss: {4:.4f}, epsilon: {5:.2f}".format(self.episode, episode_reward, mean_rewards, step, episode_loss, self.D3QN.epsilon), flush=True)

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
                        file.write("episode:{0}, reward: {1:.4f}, mean reward: {2}, total steps: {3}, epsilon: {4:.3f}, Result: {5}\n".format(self.episode, episode_reward, mean_rewards, step, self.D3QN.epsilon, terminal_state))
                    
                    self.writer.add_scalar('reward', episode_reward, self.episode)
                    self.writer.add_scalar('mean reward', mean_rewards, self.episode)
                    self.writer.add_scalar('total steps', step, self.episode)
                    self.writer.add_scalar('avg_loss', episode_loss, self.episode)

                    if self.episode % self.save_interval == 0:
                        self.D3QN.save_models(self.episode)

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
                    print()

                    break
        
        print("total_done : {} step".format(self.D3QN.eps_step))

    def test(self):
        self.D3QN.validation = True
        self.D3QN.record_path = self.checkpoint
        self.D3QN.load_models()
        for i in range(150):
            
            obs, info = self.env.reset_test() # obs [1,2,144,256] ; info [1,7] numpy
            obs = T.tensor(obs, dtype=T.float32, device=self.device)
            info = T.tensor(info, dtype=T.float32, device=self.device)
            #latent = self.cmvae(obs) # [1, latent_dim]
            step = 0
            episode_reward = 0
            done = False
            success = False
            exceed = False
            episode_loss = 0
            terminal_state = ""
            start = time.time()
            while True:
                action = self.D3QN.choose_action(obs, info)
                nx_obs, nx_info, reward, done, collision, exceed, success = self.env.step(action, step)
                nx_obs = T.tensor(nx_obs, dtype=T.float32, device=self.device)
                nx_info = T.tensor(nx_info, dtype=T.float32, device=self.device)
                #nx_latent = self.cmvae(nx_obs) # [1, latent_dim]

                episode_reward += reward
                obs = nx_obs
                info = nx_info
                #latent = nx_latent

                step += 1

                if done :
                    if collision:
                        self.collision_count += 1
                        terminal_state = "Collision"
                    elif exceed:
                        self.t_exceed_count += 1
                        terminal_state = "Time exceed"
                        
                    elif success:
                        self.success_count += 1
                        terminal_state = "Success landing"
                    
                    end = time.time()
                    stopWatch = end - start
                    print("-----------------------------------------------------", flush=True)
                    #print("episode:{0}, reward: {1:.2f}, total steps: {2}, loss: {3:.4f}, epsilon: {4:.2f}".format(self.episode, episode_reward, step, episode_loss, self.D3QN.epsilon), flush=True)
                    print("Episode is done, episode time: ", stopWatch)
                    print("Result : {}".format(terminal_state))

                    with open("test_result.txt","a")as f:
                        f.write("Episode time: {0:.2f}, Result: {1}\n".format(stopWatch,terminal_state))
                    break

class RandomAgent():
    def __init__(self):
        self.env = ueEnv()
        self.collision_count = 0
        self.t_exceed_count = 0
        self.success_count = 0

    def test(self):
        
        for i in range(150):
            obs, info = self.env.reset_test() # obs [1,2,144,256] ; info [1,7] numpy
            #obs = T.tensor(obs, dtype=T.float32, device=self.device)
            #info = T.tensor(info, dtype=T.float32, device=self.device)
            #latent = self.cmvae(obs) # [1, latent_dim]
            step = 0
            episode_reward = 0
            done = False
            success = False
            exceed = False
            episode_loss = 0
            terminal_state = ""
            start = time.time()
            while True:
                action = random.randint(0,6)
                nx_obs, nx_info, reward, done, collision, exceed, success = self.env.step(action, step)
                #nx_obs = T.tensor(nx_obs, dtype=T.float32, device=self.device)
                #nx_info = T.tensor(nx_info, dtype=T.float32, device=self.device)
                #nx_latent = self.cmvae(nx_obs) # [1, latent_dim]

                episode_reward += reward
                obs = nx_obs
                info = nx_info
                #latent = nx_latent

                step += 1

                if done :
                    if collision:
                        self.collision_count += 1
                        terminal_state = "Collision"
                    elif exceed:
                        self.t_exceed_count += 1
                        terminal_state = "Time exceed"
                        
                    elif success:
                        self.success_count += 1
                        terminal_state = "Success landing"
                    
                    end = time.time()
                    stopWatch = end - start
                    print("-----------------------------------------------------", flush=True)
                    #print("episode:{0}, reward: {1:.2f}, total steps: {2}, loss: {3:.4f}, epsilon: {4:.2f}".format(self.episode, episode_reward, step, episode_loss, self.D3QN.epsilon), flush=True)
                    print("Episode is done, episode time: ", stopWatch)
                    print("Result : {}".format(terminal_state))

                    with open("test_result.txt","a")as f:
                        f.write("Episode time: {0:.2f}, Result: {1}\n".format(stopWatch,terminal_state))
                    break

