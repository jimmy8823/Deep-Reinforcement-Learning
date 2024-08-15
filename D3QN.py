import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim
import os, math
from setuptools import glob
import torch as T
from collections import deque
#input_images_ch = 2
#batch_size = 16
#input_images (batch,channel,144,256)

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
class SumTree:
    data_pointer = 0
    def __init__(self, obs_dim, info_dim, memory_size, cm_vae):
        self.capacity = memory_size  # for all priority values
        self.cm_vae = cm_vae
        self.tree = np.zeros((2 * self.capacity - 1))
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        if self.cm_vae:
            self.obs_memory = np.zeros((self.capacity, obs_dim))
        else:
            self.obs_memory = np.zeros((self.capacity, obs_dim[1], obs_dim[2], obs_dim[3]))
        self.info_memory = np.zeros((self.capacity, info_dim))
        self.action_memory = np.zeros((self.capacity, ))
        self.reward_memory = np.zeros((self.capacity, ))
        if self.cm_vae:
            self.next_obs_memory = np.zeros((self.capacity, obs_dim))
        else:
            self.next_obs_memory = np.zeros((self.capacity, obs_dim[1], obs_dim[2], obs_dim[3]))
        self.next_info_memory = np.zeros((self.capacity, info_dim))
        self.terminal_memory = np.zeros((self.capacity, ), dtype=np.bool_)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, obs, info, action, reward, nx_obs, nx_info, done):
        tree_idx = self.data_pointer + self.capacity - 1

        self.obs_memory[self.data_pointer] = obs  # update data_frame
        self.info_memory[self.data_pointer] = info
        self.action_memory[self.data_pointer] = action
        self.reward_memory[self.data_pointer] = reward
        self.next_obs_memory[self.data_pointer] = nx_obs
        self.next_info_memory[self.data_pointer] = nx_info
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
        info = self.info_memory[data_idx]
        action = self.action_memory[data_idx]
        reward = self.reward_memory[data_idx] 
        nx_obs = self.next_obs_memory[data_idx] 
        nx_info = self.next_info_memory[data_idx] 
        done = self.terminal_memory[data_idx]
        
        return leaf_idx, self.tree[leaf_idx], obs, info, action, reward, nx_obs, nx_info, done

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

    def __init__(self, obs_dim,info_dim ,memory_size, batch_size,cm_vae):
        self.tree = SumTree(obs_dim,info_dim ,memory_size,cm_vae)
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.info_dim = info_dim
        self.warm = False
        self.cm_vae = cm_vae

    def store_transition(self, obs, info, action, reward, nx_obs, nx_info, done):
        """
        obs = obs.detach().cpu().numpy()
        info = info.detach().cpu().numpy()
        nx_obs = nx_obs.detach().cpu().numpy()
        nx_info = nx_info.detach().cpu().numpy()
        """
        max_p = np.max(self.tree.tree[-self.tree.capacity:]) # search max priority in tree
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, obs, info, action, reward, nx_obs, nx_info, done)   # set the max p for new p

    def sample_buffer(self):
        b_idx = np.empty((self.batch_size,), dtype=np.int32)

        # data frame
        if self.cm_vae:
            b_obs = np.empty((self.batch_size, self.obs_dim)) 
        else :
            b_obs = np.empty((self.batch_size, self.obs_dim[1], self.obs_dim[2], self.obs_dim[3])) 
        b_info = np.empty((self.batch_size, self.info_dim))
        b_actions = np.empty((self.batch_size,)) # action
        b_rewards =np.empty((self.batch_size,)) # reward
        if self.cm_vae:
            b_nx_obs = b_nx_obs = np.empty((self.batch_size, self.obs_dim)) # next state
        else:
            b_nx_obs = np.empty((self.batch_size, self.obs_dim[1], self.obs_dim[2], self.obs_dim[3])) # next state
        b_nx_info = np.empty((self.batch_size, self.info_dim))
        b_dones = np.empty((self.batch_size,),dtype=bool) # done 

        ISWeights = np.empty((self.batch_size, 1))

        pri_seg = self.tree.total_p / self.batch_size       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        
        for i in range(self.batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, obs, info, action, reward, nx_obs, nx_info, done = self.tree.get_leaf(v)
            prob = p / self.tree.total_p # 計算從所有優先權種中選到p的機率(for importance sampling)
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_obs[i], b_info[i], b_actions[i], b_rewards[i], b_nx_obs[i], b_nx_info[i], b_dones[i] = idx, obs, info, action, reward, nx_obs, nx_info, done
        return b_idx, b_obs, b_info, b_actions, b_rewards, b_nx_obs, b_nx_info, b_dones, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p) 

    def warming_up(self):
        if self.tree.data_pointer >= self.batch_size:
            self.warm = True
        return self.warm
    
class ReplayBuffer:
    def __init__(self, obs_dim, info_dim, memory_size, batch_size):
        self.size = memory_size
        self.batch_size = batch_size
        self.count = 0
        
        self.obs_memory = np.zeros((self.size, obs_dim[1], obs_dim[2], obs_dim[3]))
        self.info_memory = np.zeros((self.size, info_dim))
        self.action_memory = np.zeros((self.size, ))
        self.reward_memory = np.zeros((self.size, ))
        self.next_obs_memory = np.zeros((self.size, obs_dim[1], obs_dim[2], obs_dim[3]))
        self.next_info_memory = np.zeros((self.size, info_dim))
        self.terminal_memory = np.zeros((self.size, ), dtype=np.bool_)

    def store_transition(self, obs, info, action, reward, nx_obs, nx_info, done):
        # prevent index(count) exceed memory size
        idx = self.count % self.size
        self.obs_memory[idx] = obs # state
        self.info_memory[idx] = info 
        self.action_memory[idx] = action # action
        self.reward_memory[idx] = reward # reward
        self.next_obs_memory[idx] = nx_obs # next state
        self.next_info_memory[idx] = nx_info
        self.terminal_memory[idx] = done # done
        # print("store transition")
        self.count+=1
        
    def sample_buffer(self):

        # random sample a batch of transition from memory replay
        real_volume = min(self.size,self.count)
        sample = np.random.choice(real_volume, self.batch_size, replace=False)
        obs = self.obs_memory[sample] # state
        info = self.info_memory[sample]
        actions = self.action_memory[sample] # action
        rewards = self.reward_memory[sample] # reward
        nx_obs = self.next_obs_memory[sample] # next state
        nx_info = self.next_info_memory[sample]
        dones = self.terminal_memory[sample] # done

        return obs, info, actions, rewards, nx_obs, nx_info, dones
    
    def warming_up(self):
        return self.count > self.batch_size

class NStepMemory(object):
    # memory buffer to store episodic memory
    def __init__(self, memory_size=300, n_steps=4, gamma = 0.99):
        self.buffer = []
        self.memory_size = memory_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.next_idx = 0
        
    def push(self, obs, info, action, reward, next_obs, next_info, done):
        # Tensor to numpy
        obs = obs.detach().cpu().numpy()
        info = info.detach().cpu().numpy()
        next_obs = next_obs.detach().cpu().numpy()
        next_info = next_info.detach().cpu().numpy()

        data = (obs, info, action, reward, next_obs, next_info, done)
        if len(self.buffer) <= self.memory_size:    # buffer not full
            self.buffer.append(data)
        else:                                       # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def get(self, i):
        # sample episodic memory
        # [obs, info, action, reward, next_obs, next_info, done]

        begin = i
        finish = i + self.n_steps if(i + self.n_steps < self.size()) else self.size()
        sum_reward = 0 # n_step rewards
        data = self.buffer[begin:finish]
        obs = data[0][0]
        info = data[0][1]    
        action = data[0][2]
        for j in range(len(data)):
            # compute the n-th reward
            sum_reward += (self.gamma**j) * data[j][3]
            if data[j][6]:
                # manage end of episode
                next_obs = data[j][4]
                next_info = data[j][5]
                done = 1
                break
            else:
                next_obs = data[j][4]
                next_info = data[j][5]
                done = 0

        return obs, info, action, sum_reward, next_obs, next_info, done
    
    def reset(self):
        self.buffer = []
        self.next_idx = 0

    def size(self):
        return len(self.buffer)

class NoisyLinear(): # for noisy net
    def __init__(self,in_features,out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.u_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_w  = nn.Parameter(torch.Tensor(out_features, in_features))
        self.u_b = nn.Parameter(torch.Tensor(out_features))
        self.sigma_b = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # 初期値設定
        stdv = 1. / math.sqrt(self.u_w.size(1))
        self.u_w.data.uniform_(-stdv, stdv)
        self.u_b.data.uniform_(-stdv, stdv)

        initial_sigma = 0.5 * stdv
        self.sigma_w.data.fill_(initial_sigma)
        self.sigma_b.data.fill_(initial_sigma)
    
    def forward(self, x): # epsilon w = f(epsilon i)f(epislon j) epislon b = f(epislon j)
        # 毎回乱数を生成
        rand_in = self._f(torch.randn(1, self.in_features, device=self.u_w.device))
        rand_out = self._f(torch.randn(self.out_features, 1, device=self.u_w.device))
        epsilon_w = torch.matmul(rand_out, rand_in)
        epsilon_b = rand_out.squeeze()

        w = self.u_w + self.sigma_w * epsilon_w
        b = self.u_b + self.sigma_b * epsilon_b
        return F.linear(x, w, b)

    def _f(self, x): # for Factorised Gaussian noise 
       return torch.sign(x) * torch.sqrt(torch.abs(x))
    
class CNN_Net(nn.Module): # cnn net
    def __init__(self,lr,obs_dim,info_dim,action_dim,NoisyNet=False):
        super().__init__()
        # conv 1
        self.conv1 = nn.Conv2d(obs_dim[1],16,kernel_size=5,stride=2)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # conv2
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,stride=2)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        #conv3
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1)
        self.batch3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # result shape[Batch,64,7,4]
        self.fc1 = nn.Linear(1792+info_dim,128)
        self.fc2 = nn.Linear(128,64)

        self.VH = nn.Linear(64,32)
        self.AH = nn.Linear(64,32)

        self.V = nn.Linear(32,1)
        self.A = nn.Linear(32,action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self,x,info,dim=1):
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

        out = torch.flatten(out,start_dim=dim)
        out = torch.concatenate((out,info),dim=dim)

        out = self.fc1(out)
        out = self.fc2(out)

        vh = F.relu(self.VH(out))
        ah = F.relu(self.AH(out))

        V = self.V(vh)
        A = self.A(ah)
        Q = V + A - torch.mean(A,dim=1,keepdim=True)
        return Q
        
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

class Noisy_Net(nn.Module):
    def __init__(self,lr,obs_dim,info_dim,action_dim,NoisyNet=False):
        super().__init__()

        self.fc1 = NoisyLinear(obs_dim+info_dim,128)
        self.fc2 = NoisyLinear(128,64)

        # value function & advantage function(Dueling)
        self.V = NoisyLinear(64,1)
        self.A = NoisyLinear(64,action_dim)
        print("using noisy Linear !")
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.to(device)

    def forward(self,x):
        x = self.relu1(self.fc1(x))
        h = self.relu2(self.fc2(x))
        V = self.V(h)
        A = self.A(h) 
        Q = V + A - torch.mean(A,dim=-1,keepdim=True)
        return Q

class D3QN_cnn():
    
    def __init__(self, obs_dim, info_dim, action_dim, lr, gamma, epsilon, update_freq, steps, path):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.q_net = CNN_Net(lr, obs_dim, info_dim, action_dim).to(self.device) # cat your input before forward
        self.target_q_net = CNN_Net(lr, obs_dim, info_dim, action_dim).to(self.device) # cat your input before forward
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
        self.eps_decay = 20000
        self.validation = False
        
        print("Start CNN D3QN")
        #self.count_params(self.q_net)
        #summary(self.q_net, (2, 144, 256))

    def choose_action(self, obs, uav_info):
        with torch.no_grad():
            q_value = self.q_net(obs,uav_info)

        if self.validation:
            action = q_value.cpu().detach().argmax(dim=1).item()
            return action
        
        self.epsilon = self.min_eps + (self.init_eps - self.min_eps) * math.exp(-1.0 * self.eps_step / self.eps_decay)
        self.eps_step+=1

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
            print("[+] Update from target network")
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.update_count += 1
    
    def save_models(self, episode):
        checkpoint = {
            "net": self.q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "eps_step": self.eps_step
        }
        torch.save(checkpoint, ".//{0}//model//D3QN_{1}.pth".format(self.record_path,episode))
    
    def load_models(self):

        files = glob.glob('.//{}//model//*.pth'.format(self.record_path))
        files.sort(key=os.path.getmtime)
        if len(files) > 0:
            print("[+] Loading D3QN " + files[-1] + " ...")
            checkpoint = torch.load(files[-1])
            self.q_net.load_state_dict(checkpoint["net"])
            self.target_q_net.load_state_dict(checkpoint["net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.eps_step = checkpoint["eps_step"]

    def count_params(model):
        size_model = 0
        for param in model.parameters():
            if param.data.is_floating_point():
                size_model += param.numel() * torch.finfo(param.data.dtype).bits
            else:
                size_model += param.numel() * torch.iinfo(param.data.dtype).bits
            print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")

class QNET(nn.Module): # cat your input before forward
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,128)
        self.fc2 = nn.Linear(128,64)

        self.VH = nn.Linear(64,32)
        self.AH = nn.Linear(64,32)

        self.V = nn.Linear(32,1)
        self.A = nn.Linear(32,action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        h = F.relu(self.fc2(x))

        vh = F.relu(self.VH(h))
        ah = F.relu(self.AH(h))

        V = self.V(vh)
        A = self.A(ah)
        Q = V + A - torch.mean(A,dim=1,keepdim=True)
        return Q

class D3QN():
    def __init__(self, input_dim, action_dim, lr, gamma, epsilon, update_freq, steps, path,validation=False):
        super(D3QN, self).__init__()
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
        self.eps_decay = 20000
        self.validation = validation
        
    def choose_action(self, latent, uav_info):
        x = torch.cat((latent,uav_info),dim=1)
        with torch.no_grad():
            q_value = self.q_net(x)

        if self.validation:
            action = q_value.cpu().detach().argmax(dim=1).item()
            return action
        
        self.epsilon = self.min_eps + (self.init_eps - self.min_eps) * math.exp(-1.0 * self.eps_step / self.eps_decay)
        self.eps_step+=1

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
            print("[+] Update from target network")
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.update_count += 1
    
    def save_models(self, episode):
        checkpoint = {
            "net": self.q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "eps_step": self.eps_step
        }
        torch.save(checkpoint, ".//{0}//model//D3QN_{1}.pth".format(self.record_path,episode))
    
    def load_models(self):

        files = glob.glob('.//{}//model//*.pth'.format(self.record_path))
        files.sort(key=os.path.getmtime)
        if len(files) > 0:
            print("[+] Loading D3QN " + files[-1] + " ...")
            checkpoint = torch.load(files[-1])
            self.q_net.load_state_dict(checkpoint["net"])
            self.target_q_net.load_state_dict(checkpoint["net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.eps_step = checkpoint["eps_step"]
            
"""
pr = Prioritized_replay(128,7,1000,2,True)
b_idx, b_latent, b_info, b_action, b_rewards, b_nx_latent, b_nx_info, b_dones, ISWeight = pr.sample_buffer()
b_dones = [[True],[True]]

b_latent = T.tensor(np.array(b_latent, dtype=np.float32), dtype=T.float)
b_info = T.tensor(np.array(b_info, dtype=np.float32), dtype=T.float)
b_action = T.tensor(np.array(b_action, dtype=np.float32), dtype=T.float).unsqueeze(1)
b_rewards = T.tensor(np.array(b_rewards, dtype=np.float32), dtype=T.float).unsqueeze(1)
b_nx_latent = T.tensor(np.array(b_nx_latent, dtype=np.float32), dtype=T.float)
b_nx_info = T.tensor(np.array(b_nx_info, dtype=np.float32), dtype=T.float)
b_dones = T.tensor(np.array(b_dones, dtype=np.float32), dtype=T.float)
b_ISWeight = T.tensor(np.array(b_dones, dtype=np.float32), dtype=T.float)

print("b_idx:", b_idx.shape)
print("b_latent:", b_latent.shape)
print("b_info:", b_info.shape)
print("b_action:", b_action.shape)
print("b_rewards:", b_rewards.shape)
print("b_nx_latent:", b_nx_latent.shape)
print("b_nx_info:", b_nx_info.shape)
print("b_dones:", b_dones.shape)

#ction = torch.randn(2,13)
max_a = T.argmax(b_action,dim=1,keepdim=True)
print("max:", max_a.shape)
print("gather:",b_action.gather(1,max_a))
"""