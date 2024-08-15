import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from env import ueEnv
import random
import datetime
import airsim
from CM_VAE.CM_VAE_ver2 import CmVAE
import torch as T
import cv2
from tqdm import tqdm
import math
import collections

class SumTree:
    data_pointer = 0
    def __init__(self, obs_dim, memory_size, cm_vae):
        self.capacity = memory_size  # for all priority values
        self.cm_vae = cm_vae
        self.tree = np.zeros((2 * self.capacity - 1))
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        if self.cm_vae:
            self.obs_memory = np.zeros((self.capacity, obs_dim))
        else:
            self.obs_memory = np.zeros((self.capacity, 4))
        self.action_memory = np.zeros((self.capacity, ))
        self.reward_memory = np.zeros((self.capacity, ))
        if self.cm_vae:
            self.next_obs_memory = np.zeros((self.capacity, obs_dim))
        else:
            self.next_obs_memory = np.zeros((self.capacity, 4))
        self.terminal_memory = np.zeros((self.capacity, ), dtype=np.bool_)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, obs, action, reward, nx_obs, done):
        tree_idx = self.data_pointer + self.capacity - 1

        self.obs_memory[self.data_pointer] = obs  # update data_frame
        self.action_memory[self.data_pointer] = action
        self.reward_memory[self.data_pointer] = reward
        self.next_obs_memory[self.data_pointer] = nx_obs
        self.terminal_memory[self.data_pointer] = done

        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx] #計算欲更改的葉節點之priotrity變化
        self.tree[tree_idx] = p # 修改
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2 # 更新父節點
            self.tree[tree_idx] += change
    
    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        obs = self.obs_memory[data_idx] # get data
        action = self.action_memory[data_idx]
        reward = self.reward_memory[data_idx] 
        nx_obs = self.next_obs_memory[data_idx] 
        done = self.terminal_memory[data_idx]
        
        return leaf_idx, self.tree[leaf_idx], obs, action, reward, nx_obs, done

    @property
    def total_p(self):
        return self.tree[0]  # the root
class Prioritized_replay:
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, obs_dim ,memory_size, batch_size,cm_vae):
        self.tree = SumTree(obs_dim ,memory_size,cm_vae)
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.warm = False
        self.cm_vae = cm_vae

    def store_transition(self, obs, action, reward, nx_obs, done):
        obs = obs.detach().cpu().numpy()

        max_p = np.max(self.tree.tree[-self.tree.capacity:]) # search max priority in tree
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, obs, action, reward, nx_obs, done)   # set the max p for new p

    def sample_buffer(self):
        b_idx = np.empty((self.batch_size,), dtype=np.int32)

        # data frame
        if self.cm_vae:
            b_obs = np.empty((self.batch_size, self.obs_dim)) 
        else :
            b_obs = np.empty((self.batch_size, 4)) 
        b_actions = np.empty((self.batch_size,)) # action
        b_rewards =np.empty((self.batch_size,)) # reward
        if self.cm_vae:
            b_nx_obs = b_nx_obs = np.empty((self.batch_size, self.obs_dim)) # next state
        else:
            b_nx_obs = np.empty((self.batch_size, 4)) # next state
        b_dones = np.empty((self.batch_size,),dtype=bool) # done 

        ISWeights = np.empty((self.batch_size, 1))

        pri_seg = self.tree.total_p / self.batch_size       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        
        for i in range(self.batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, obs, action, reward, nx_obs, done = self.tree.get_leaf(v)
            prob = p / self.tree.total_p # 計算從所有優先權種中選到p的機率(for importance sampling)
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_obs[i], b_actions[i], b_rewards[i], b_nx_obs[i], b_dones[i] = idx, obs, action, reward, nx_obs, done
        return b_idx, b_obs, b_actions, b_rewards, b_nx_obs, b_dones, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):            self.tree.update(ti, p) 

    def warming_up(self):
        if self.tree.data_pointer >= self.batch_size:
            self.warm = True
        return self.warm
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        state = state.detach().cpu().numpy()
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
class NStepMemory(object):
    # memory buffer to store episodic memory
    def __init__(self, memory_size=300, n_steps=4, gamma = 0.99):
        self.buffer = []
        self.memory_size = memory_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.next_idx = 0
        
    def push(self, obs, action, reward, next_obs, done):
        # Tensor to numpy
        data = (obs, action, reward, next_obs, done)
        if len(self.buffer) <= self.memory_size:    # buffer not full
            self.buffer.append(data)
        else:                                       # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def get(self, i):
        # sample episodic memory
        # [obs, action, reward, next_obs, done]

        begin = i
        finish = i + self.n_steps if(i + self.n_steps < self.size()) else self.size()
        sum_reward = 0 # n_step rewards
        data = self.buffer[begin:finish]
        obs = data[0][0]
        action = data[0][1]
        for j in range(len(data)):
            # compute the n-th reward
            sum_reward += (self.gamma**j) * data[j][2]
            if data[j][4]:
                # manage end of episode
                next_obs = data[j][3]
                done = 1
                break
            else:
                next_obs = data[j][3]
                done = 0

        return obs, action, sum_reward, next_obs, done
    
    def reset(self):
        self.buffer = []
        self.next_idx = 0

    def size(self):
        return len(self.buffer)
    
class QNET(nn.Module): # cat your input before forward
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,256)
        self.fc2 = nn.Linear(256,128)

        self.v_h = nn.Linear(128,64)
        self.a_h = nn.Linear(128,64)

        self.V = nn.Linear(64,1)
        self.A = nn.Linear(64,action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        h = F.relu(self.fc2(x))

        vh = self.v_h(h)
        va = self.a_h(h)
        V = self.V(vh)
        A = self.A(va)
        Q = V + A - torch.mean(A,dim=1,keepdim=True)
        return Q
    
class D3QN():
    def __init__(self, input_dim, action_dim, lr, gamma, epsilon, update_freq, steps, path):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.q_net = QNET(input_dim,action_dim).to(self.device) # cat your input before forward
        self.target_q_net = QNET(input_dim,action_dim).to(self.device) # cat your input before forward
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = lr)
        self.record_path = path
        self.action_dim = action_dim
        self.eps_step = 0 # doing how many step
        self.update_count = 0 # count for update target network
        self.gamma = gamma 
        self.tgr_update_freq = update_freq
        self.epsilon = epsilon
        self.init_eps = 1.0
        self.min_eps = 0.05
        self.eps_decay = 50000

    def choose_action(self, x):
        with torch.no_grad():
            q_value = self.q_net(x)

        self.epsilon = 0.01

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = q_value.cpu().detach().argmax(dim=1).item()

        return action
    
    def update(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.update_count % self.tgr_update_freq == 0:
            #print("[+] Update from target network")
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.update_count += 1
    

"""class D3QN(nn.Module):
    def __init__(self,):
        super().__init__()
        # conv 1
        self.conv1 = nn.Conv2d(input_images_ch,16,kernel_size=5,stride=2)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # conv2
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=2)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        #conv3
        self.conv3 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # result shape[16,32,7,5]
        #FC
        self.fc1 = nn.Linear(17920,128)
        self.fc2 = nn.Linear(128,13)
    def forward(self,x):
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.batch3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = torch.flatten(out)

        out = self.fc1(out)
        out = self.fc2(out)
        return out
"""
def test_env():
    env = ueEnv()
    obs, info = env.reset()
    action = random.randint(0,12)
    done = False
    collision = False
    exceed = False
    success = False
    step = 0 
    while not done :
        step+=1
        action = 1 #random.randint(0,12) # forward 
        nx_obs, nx_info, reward, done, collision, exceed, success = env.step(obs,info,action,step)
        #print("gray : ", obs[0][0])
        #print("depth min: ", np.min(obs[0][1]))
        #print("info : ", info[6])
        
        # store transition
        obs = nx_obs
        info = nx_info
    
    print("collision : ", collision)
    print("exceed : ", exceed)
    print("success : ", success)

def test_vae_realtime():
    env = ueEnv()

    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
    cmvae_path = "D:/CODE/Python/AirSim/CM_VAE/mix_origin_depth/cmvae_50.pth"#"D:/CODE/Python/AirSim/CM_VAE/mix_noise/cmvae_40.pth"
    cmvae = CmVAE(input_dim=2,semantic_dim=1)
    cmvae.load_state_dict(T.load(cmvae_path))
    cmvae.eval()
    obs, info = env.reset() # obs [1,2,144,256] ; info [1,7] numpy
    obs = T.tensor(obs, dtype=T.float32, device=device)
    info = T.tensor(info, dtype=T.float32, device=device)
    test_idx = 0
    threshold = 0.5
    total_reward = 0
    while 1:
        reward = 0
        key = (airsim.wait_key('Press any key')).decode("utf-8")
        if(key == 'w'):
            nx_obs, nx_info, reward, done, collision, exceed, success = env.step(1,0)
            nx_obs = T.tensor(nx_obs, dtype=T.float32, device=device)
            nx_info = T.tensor(nx_info, dtype=T.float32, device=device)
            
        elif(key == 'a'):
            nx_obs, nx_info, reward, done, collision, exceed, success = env.step(4,0)
            nx_obs = T.tensor(nx_obs, dtype=T.float32, device=device)
            nx_info = T.tensor(nx_info, dtype=T.float32, device=device)
        elif(key == 'd'):
            nx_obs, nx_info, reward, done, collision, exceed, success = env.step(2,0)
            nx_obs = T.tensor(nx_obs, dtype=T.float32, device=device)
            nx_info = T.tensor(nx_info, dtype=T.float32, device=device)
        elif(key == '0'):
            nx_obs, nx_info = env.reset()
            nx_obs = T.tensor(nx_obs, dtype=T.float32, device=device)
            nx_info = T.tensor(nx_info, dtype=T.float32, device=device)
            total_reward = 0
        elif(key == "e"):
            nx_obs, nx_info, reward, done, collision, exceed, success = env.step(6,0)
            nx_obs = T.tensor(nx_obs, dtype=T.float32, device=device)
            nx_info = T.tensor(nx_info, dtype=T.float32, device=device)
        elif(key == "o"):
            test_idx += 1
            reconstruction,_ = cmvae(obs)
            reconstruction[reconstruction>=threshold]=255
            reconstruction[reconstruction<threshold]=0
            reconstruction = reconstruction.detach().squeeze().cpu().numpy()
            gray = obs[0][0].detach().cpu().numpy()
            cv2.imshow('Original Image', gray)
            cv2.imshow('Noisy Image', reconstruction)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            #airsim.write_png(f"D:/CODE/Python/AirSim/CM_VAE/reconstruction_cmvae_{test_idx}.png",reconstruction)
        elif(key == "p"):
            break
        
        total_reward += reward
        obs = nx_obs
        info = nx_info
        if collision :
            nx_obs, nx_info = env.reset()
            nx_obs = T.tensor(nx_obs, dtype=T.float32, device=device)
            nx_info = T.tensor(nx_info, dtype=T.float32, device=device)
            total_reward = 0
            
        print("Reward : {0:.4f} , Total Reward : {1:.4f} , info attd : {2:.4f}".format(reward, total_reward, nx_info[0][6]))

def test_D3QN_cartpole():
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    random.seed(0)
    np.random.seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    lr = 0.0002
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 200
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
    agent = D3QN(state_dim, action_dim, lr, gamma, epsilon,
            target_update, 0, "")
    memory = Prioritized_replay(obs_dim=4,memory_size=buffer_size, batch_size=batch_size,cm_vae=False)
    nsmemory = NStepMemory()
    #ReplayBuffer(buffer_size)
    
    return_list = []

    for i in range(100):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state,_ = env.reset()
                done = False
                step = 0
                while not done:
                    state = T.tensor(np.array(state, dtype=np.float32), dtype=T.float, device=device).unsqueeze(0)
                    action = agent.choose_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    state = state.squeeze(0)
                    nsmemory.push(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    step += 1
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                for s in range(step):
                    obs, action, reward, nx_obs, done = nsmemory.get(s)
                    memory.store_transition(obs, action, reward, nx_obs, done)
                nsmemory.reset()
                if memory.warming_up():
                    b_idx, b_s, b_a, b_r, b_ns, b_d, b_is = memory.sample_buffer()
                    b_s = T.tensor(np.array(b_s, dtype=np.float32), dtype=T.float, device=device)
                    b_a = T.tensor(np.array(b_a, dtype=np.float32), dtype=T.int64, device=device).unsqueeze(1)
                    b_r = T.tensor(np.array(b_r, dtype=np.float32), dtype=T.float, device=device).unsqueeze(1)
                    b_ns = T.tensor(np.array(b_ns, dtype=np.float32), dtype=T.float, device=device)
                    b_d = T.tensor(np.array(b_d, dtype=np.float32), dtype=T.float, device=device).unsqueeze(1)
                    b_is = T.tensor(np.array(b_is, dtype=np.float32), dtype=T.float,  device=device).unsqueeze(1)
                    with T.no_grad():
                        next_q_value = agent.target_q_net(b_ns) # target net 進行估計target value
                        max_action = T.argmax(agent.q_net(b_ns),dim=1,keepdim=True) # double network (eval net 估計最大Q value作為選擇的action)
                        q_targets = b_r + gamma * next_q_value.gather(1,max_action) * (1 - b_d)
                    q_values = agent.q_net(b_s).gather(1,b_a)

                    abs_errors = T.abs(q_values - q_targets).detach().cpu().numpy() #計算TD error

                    memory.batch_update(b_idx, abs_errors) #更新priority

                    loss = (b_is*F.mse_loss(q_values, q_targets.detach())).mean()
                    agent.update(loss)

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

def test():
    x = "episode:2000, reward: 13.4155, mean reward: 0.23, total steps: 59, epsilon: 0.056, Result: Success landing"
    y = int(x.split(",")[0].split(":")[1]) +1
    print(y)

test_vae_realtime()