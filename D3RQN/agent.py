import os, sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import time
import cv2
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter 
from tqdm import tqdm

from VAE.vae import VAE
from env import ueEnv
from D3RQN import D3RQN
from buffer import NStepMemory, Buffer

device = T.device("cuda") if T.cuda.is_available() else T.device("cpu")
T.backends.cudnn.benchmark = True

class Agent():
    def __init__(self):
        self.max_episodes = 3000
        self.save_interval = 50
        self.episode = -1
        self.max_steps = 300
        self.batch_size = 16
        self.n_steps = 4
        self.gamma = 0.99
        self.memory_size = 2000
        self.collision_count = 0
        self.t_exceed_count = 0
        self.success_count = 0
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        with open('log.txt') as f:
            last_line = f.readlines()
            total_step = 0
            for li in last_line:
                total_step += int(li.strip('\n').split(',')[4].split(":")[1])

            if len(last_line) != 0:
                last_line = last_line[-1]
                self.episode = int(last_line.strip('\n').split(',')[0].split(":")[1]) + 1
                print("[+] Start with: {0}".format(self.episode), flush=True)

        self.env = ueEnv()

        self.D3RQN = D3RQN(action_dim=7, learning_rate=1e-4, gamma=self.gamma, epsilon=0.999, target_update=5000, steps=total_step)
        self.vae = VAE(1,8192,64).to(self.device)
        
        print("[+] Loading VAE ...", flush=True)
        self.vae.load_state_dict(T.load("D:\\CODE\\Python\\AirSim\\D3RQN\\VAE\\vae_100.pth"))
        self.vae.eval()
        self.D3RQN.load_models()

        self.n_steps_memory = NStepMemory(self.max_steps, self.n_steps, self.gamma)
        self.memory = Buffer(memory_size = self.memory_size, burn_in = 8, a = 0.6, e = 0.01)
        

        if T.cuda.is_available():
            print('[+] Using device:', T.cuda.get_device_name(0), flush=True)
        else:
            print("Using CPU")
        
        
    def learn(self):
        if self.memory.size() < self.batch_size:
            return 0

        idxs, state_grays, ac_states, actions, rewards, next_state_grays, next_ac_states, dones, hs, cs, next_hs, next_cs = self.memory.sample(self.batch_size)

        state_grays = T.tensor(np.array(state_grays, dtype=np.float32), dtype=T.float, device=self.device).squeeze(1)
        ac_states = T.tensor(np.array(ac_states, dtype=np.float32), dtype=T.float, device=self.device).squeeze(1)
        actions =  T.tensor(np.array(actions, dtype=np.float32), dtype=T.int64, device=self.device).view([-1, 1, 1])
        rewards = T.tensor(np.array(rewards, dtype=np.float32), dtype=T.float, device=self.device).view([-1, 1, 1])
        next_state_grays = T.tensor(np.array(next_state_grays, dtype=np.float32), dtype=T.float, device=self.device).squeeze(1)
        next_ac_states = T.tensor(np.array(next_ac_states, dtype=np.float32), dtype=T.float, device=self.device).squeeze(1)
        dones = T.tensor(np.array(dones, dtype=np.float32), dtype=T.float, device=self.device).view([-1, 1, 1])
        h = T.tensor(np.array(hs, dtype=np.float32), dtype=T.float, device=self.device).squeeze(1).transpose(0, 1)
        c = T.tensor(np.array(cs, dtype=np.float32), dtype=T.float, device=self.device).squeeze(1).transpose(0, 1)
        next_h = T.tensor(np.array(next_hs, dtype=np.float32), dtype=T.float, device=self.device).squeeze(1).transpose(0, 1)
        next_c = T.tensor(np.array(next_cs, dtype=np.float32), dtype=T.float, device=self.device).squeeze(1).transpose(0, 1)

        q_value, _, _ = self.D3RQN.q_net(state_grays, ac_states, h, c)
        q_values = q_value.gather(2, actions)

        max_action, _, _ = self.D3RQN.q_net(next_state_grays, next_ac_states, next_h, next_c)
        max_actions = max_action.max(2)[1].view(-1, 1, 1)

        next_q_value, _, _ = self.D3RQN.target_q_net(next_state_grays, next_ac_states, next_h, next_c)
        max_next_q_values = next_q_value.gather(2, max_actions).detach()

        q_targets = rewards + (self.D3RQN.gamma ** self.n_steps) * max_next_q_values * (1 - dones)
        errors = (q_values - q_targets).detach().cpu().squeeze().tolist()
        self.memory.update(idxs, errors)

        loss = T.mean(F.mse_loss(q_values, q_targets))
        self.D3RQN.update(loss)
        return loss

    def show_image(self, img, recon_img):
        img = np.uint8(img)
        recon_img = np.squeeze(recon_img.cpu().detach().numpy(), (0, 1))
        recon_img = recon_img * 255
        recon_img = np.uint8(recon_img)

        cv2.imshow("img", img)
        cv2.imshow("recon_img", recon_img)
        cv2.waitKey(1)
        
    def transformToTensor(self, img, img_type):
        '''Transform numpy image to Tensor

        Args:
            img: transfer image.
            img_type: transfer type. 0 or 1
        Return:
            return tensor data.
            example shape:
                img_type: 0 (1, 1, 240, 320)
                img_type: 1 (1, 240, 320)

        '''
        if(img_type == 0): #Depth
            tensor = T.tensor(img, dtype=T.float, device=self.device)
            tensor = tensor.unsqueeze(0)
            tensor = tensor.unsqueeze(0)
            tensor = (tensor / 255).float()
        else: #Gray
            tensor = T.tensor(img, dtype=T.float, device=self.device)
            tensor = tensor.unsqueeze(0)
            tensor = (tensor / 255).float()
        return tensor

    def train(self):
        print("[+] Start Training...", flush=True)

        if self.episode == -1:
            self.episode = 1
        
        self.writer = SummaryWriter('logs/D3RQN')

        while(self.episode <= self.max_episodes):
            start = time.time()

            obs, info = self.env.reset()
            info = T.tensor(info, dtype=T.float32, device=self.device)       

            state_depth = obs[0][1] # (144, 256)
            state_gray = obs[0][0]   # (144, 256)
            state_depth = np.expand_dims(state_depth, 0)
            state_depth = np.expand_dims(state_depth, 0) # (1, 1, 144, 256)
            state_gray = np.expand_dims(state_gray, 0) # (1, 144, 256)
            
            state_gray = T.tensor(state_gray, dtype=T.float32).to(device)
            state_depth = T.tensor(state_depth,dtype=T.float32).to(device)
            
            ac_state = self.vae.val(state_depth)               # (1, 64)
            ac_state = T.cat((ac_state, info), dim=1)        # (1, 71)

            # 8 state
            all_state_gray = T.stack((state_gray, state_gray, state_gray, state_gray, state_gray, state_gray, state_gray, state_gray), dim=1)   # (1, 8, 144, 256)
            all_ac_state = T.stack((ac_state, ac_state, ac_state, ac_state, ac_state, ac_state, ac_state, ac_state), dim=1)                     # (1, 8, 71)
            steps = 0
            score = 0

            h, c = self.D3RQN.init_hidden_state(batch_size=self.batch_size, training=False)

            while True:
                action, next_h, next_c = self.D3RQN.choose_action(all_state_gray, all_ac_state, h, c)
                
                nx_obs, nx_info, reward, done, collision, exceed, success = self.env.step(action, steps)
                
                # (1, 144, 256)
                next_state_depth = T.tensor(nx_obs[0][1],dtype=T.float32, device=self.device).unsqueeze(0)
                # (1, 144, 256)
                next_state_gray = T.tensor(nx_obs[0][0], dtype=T.float32, device=self.device).unsqueeze(0)
                
                next_state_depth = next_state_depth.unsqueeze(0) # (1, 1, 144, 256)
                next_state_gray = next_state_gray.unsqueeze(0) # (1, 1, 144, 256)

                nx_info = T.tensor(nx_info, dtype=T.float32, device=self.device)

                next_ac_state = self.vae.val(next_state_depth)
                
                next_ac_state = T.cat((next_ac_state, nx_info), dim=1)
                next_all_ac_state = T.cat((all_ac_state[:, 1:], next_ac_state.unsqueeze(0)), dim=1)
                next_all_state_gray = T.cat((all_state_gray[:, 1:], next_state_gray), dim=1)

                if steps == self.max_steps: 
                    done = 1
                else:
                    steps += 1


                self.n_steps_memory.push(all_state_gray, all_ac_state, action, reward, next_all_state_gray, next_all_ac_state, done, h, c, next_h, next_c)

                h, c = next_h, next_c
                all_ac_state = next_all_ac_state
                all_state_gray = next_all_state_gray
                score += reward

                if done:
                    if steps < 5:
                        break

                    # for s in range(steps - self.n_steps + 1):
                    for s in range(steps):
                        self.memory.push(self.n_steps_memory.get(s))
                    self.n_steps_memory.reset()

                    totol_loss = 0
                    for t in tqdm(range(0, steps)):
                        loss = self.learn()
                        totol_loss += loss

                    totol_loss /= steps

                    
                    mean_rewards = round(score / steps, 2)
                    print("-----------------------------------------------------", flush=True)
                    print("episode:{0}, reward: {1}, mean reward: {2}, score: {3}, total steps: {4}, loss: {5}, epsilon: {6}, memory: {7}".format(self.episode, reward, mean_rewards, score, steps, totol_loss, self.D3RQN.epsilon, self.memory.tree.write), flush=True)

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

                    with open('log.txt', 'a') as file:
                        file.write("episode:{0}, reward: {1}, mean reward: {2}, score: {3}, total steps: {4}\n".format(self.episode, reward, mean_rewards, score, steps))
                    
                    self.writer.add_scalar('reward', score, self.episode)
                    self.writer.add_scalar('mean reward', mean_rewards, self.episode)
                    self.writer.add_scalar('total steps', steps, self.episode)
                    self.writer.add_scalar('avg_loss', totol_loss, self.episode)
                    
                    if self.episode % self.save_interval == 0:
                        self.D3RQN.save_models(self.episode)

                    if self.episode % 50 == 0: # record terminal state per 50 episode
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

    def test(self):
        self.D3RQN.validation = True

        for i in range(150):
            obs, info = self.env.reset_test() # obs [1,2,144,256] ; info [1,7] numpy

            info = T.tensor(info, dtype=T.float32, device=self.device)   
            state_depth = obs[0][1] # (144, 256)
            state_gray = obs[0][0]   # (144, 256)
            state_depth = np.expand_dims(state_depth, 0)
            state_depth = np.expand_dims(state_depth, 0) # (1, 1, 144, 256)
            state_gray = np.expand_dims(state_gray, 0) # (1, 144, 256)

            state_gray = T.tensor(state_gray, dtype=T.float32).to(device)
            state_depth = T.tensor(state_depth,dtype=T.float32).to(device)
            
            ac_state = self.vae.val(state_depth)               # (1, 64)
            ac_state = T.cat((ac_state, info), dim=1)        # (1, 71)

            # 8 state
            all_state_gray = T.stack((state_gray, state_gray, state_gray, state_gray, state_gray, state_gray, state_gray, state_gray), dim=1)   # (1, 8, 144, 256)
            all_ac_state = T.stack((ac_state, ac_state, ac_state, ac_state, ac_state, ac_state, ac_state, ac_state), dim=1)                     # (1, 8, 71)
            
            terminal_state = ""
            steps = 0
            score = 0

            h, c = self.D3RQN.init_hidden_state(batch_size=self.batch_size, training=False)
            start = time.time()
            while True:
                action, next_h, next_c = self.D3RQN.choose_action(all_state_gray, all_ac_state, h, c)
                
                nx_obs, nx_info, reward, done, collision, exceed, success = self.env.step(action, steps)
                
                # (1, 144, 256)
                next_state_depth = T.tensor(nx_obs[0][1],dtype=T.float32, device=self.device).unsqueeze(0)
                # (1, 144, 256)
                next_state_gray = T.tensor(nx_obs[0][0], dtype=T.float32, device=self.device).unsqueeze(0)
                
                next_state_depth = next_state_depth.unsqueeze(0) # (1, 1, 144, 256)
                next_state_gray = next_state_gray.unsqueeze(0) # (1, 1, 144, 256)

                nx_info = T.tensor(nx_info, dtype=T.float32, device=self.device)

                next_ac_state = self.vae.val(next_state_depth)
                
                next_ac_state = T.cat((next_ac_state, nx_info), dim=1)
                next_all_ac_state = T.cat((all_ac_state[:, 1:], next_ac_state.unsqueeze(0)), dim=1)
                next_all_state_gray = T.cat((all_state_gray[:, 1:], next_state_gray), dim=1)


                h, c = next_h, next_c
                all_ac_state = next_all_ac_state
                all_state_gray = next_all_state_gray

                if steps == self.max_steps: 
                    done = 1
                else:
                    steps += 1

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