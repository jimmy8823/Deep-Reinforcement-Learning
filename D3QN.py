import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim
import math
#input_images_ch = 2
#batch_size = 16
#input_images (batch,channel,144,256)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SumTree:
    data_pointer = 0
    def __init__(self, obs_dim, info_dim, memory_size):
        self.capacity = memory_size  # for all priority values
        self.tree = np.zeros((2 * self.capacity - 1))
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.obs_memory = np.zeros((self.capacity, obs_dim[1], obs_dim[2], obs_dim[3]))
        self.info_memory = np.zeros((self.capacity, info_dim))
        self.action_memory = np.zeros((self.capacity, ))
        self.reward_memory = np.zeros((self.capacity, ))
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

    def __init__(self, obs_dim,info_dim ,memory_size, batch_size):
        self.tree = SumTree(obs_dim,info_dim ,memory_size)
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.info_dim = info_dim
        self.warm = False

    def store_transition(self, obs, info, action, reward, nx_obs, nx_info, done):
        max_p = np.max(self.tree.tree[-self.tree.capacity:]) # search max priority in tree
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, obs, info, action, reward, nx_obs, nx_info, done)   # set the max p for new p

    def sample_buffer(self):
        b_idx = np.empty((self.batch_size,), dtype=np.int32)

        # data frame
        b_obs = np.empty((self.batch_size, self.obs_dim[1], self.obs_dim[2], self.obs_dim[3])) 
        b_info = np.empty((self.batch_size, self.info_dim))
        b_actions = np.empty((self.batch_size,)) # action
        b_rewards =np.empty((self.batch_size,)) # reward
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
        if self.tree.data_pointer >= 16:
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
    
class Net(nn.Module):
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
        self.conv3 = nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        # result shape[Batch,32,7,4]

        if NoisyNet:
            #FC 896 + 7(info)
            self.fc1 = NoisyLinear(896+info_dim,128)
            self.fc2 = NoisyLinear(128,64)

            # value function & advantage function(Dueling)
            self.V = NoisyLinear(64,1)
            self.A = NoisyLinear(64,action_dim)
            print("using noisy Linear !")
        else:
            self.fc1 = nn.Linear(896+info_dim,128)
            self.fc2 = nn.Linear(128,64)

            self.V = nn.Linear(64,1)
            self.A = nn.Linear(64,action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    def forward(self,x,info,dim=0):
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
        V = self.V(out)
        A = self.A(out)
        Q = V + A - torch.mean(A,dim=-1,keepdim=True)
        return Q
        
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

class D3QN():
    def __init__(self, lr, obs_dim, info_dim, action_dim, gamma=0.99, tau=0.005,
                 epsilon=1.0, eps_end=0.01, eps_dec=5e-7, memory_size=5000,
                 batch_size=16, ckpt_dir="",Prioritized_rp=False,NoisyNet=False):
        super().__init__()
        self.gamma = gamma #discount factor
        self.tau = tau #
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.checkpoint_dir = ckpt_dir # checkpoint save 
        self.batch_size = batch_size
        self.action_space = action_dim
        self.prioritized = Prioritized_rp
        self.noisyNet = NoisyNet

        self.eval_net = Net(lr, obs_dim, info_dim, action_dim,self.noisyNet)
        self.target_net = Net(lr, obs_dim, info_dim, action_dim,self.noisyNet)
        if self.prioritized:
            self.memory = Prioritized_replay(obs_dim,info_dim ,memory_size, batch_size)
            print("Start D3QN with prioritized replay! ")
        else:
            self.memory = ReplayBuffer(obs_dim,info_dim ,memory_size, batch_size)
            print("Start D3QN with native replay! ")

    def soft_update(self, tau=None):#軟更新，在更新target時保持一定比率的
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.target_net.parameters(), self.eval_net.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)
        # print("soft update")

    def hard_update(self): # 過幾回合後將eval網路權重複製給target網路
        self.target_net.load_state_dict(self.eval_net.state_dict())
        # print("hard update")
    
    def store_transition(self, obs, info, action, reward, nx_obs, nx_info, done):
        self.memory.store_transition(obs, info, action, reward, nx_obs, nx_info, done)

    def choose_action(self, observation,info, isTrain=True): # forward network
        obs = torch.from_numpy(observation).to(device)
        info = torch.from_numpy(info).to(device)
        q_values = self.eval_net.forward(obs,info) # get each action`s Q value
        action = torch.argmax(q_values).item()# 加入item()才可以拿到index 否則會回傳如tensor(5)

        if(np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.action_space)
        return action

    def reduce_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self, episode):
        self.eval_net.save_checkpoint(self.checkpoint_dir + '/eval/D3QN_q_eval_{}.pth'.format(episode))
        print("Saving eval_net success!")
        self.target_net.save_checkpoint(self.checkpoint_dir + '/target/D3QN_q_target_{}.pth'.format(episode))
        print("Saving target_net success!")

    def load_models(self, episode):
        self.eval_net.load_checkpoint(self.checkpoint_dir + 'Q_eval/D3QN_q_eval_{}.pth'.format(episode))
        print("Loading eval_net success!")
        self.target_net.load_checkpoint(self.checkpoint_dir + 'Q_target/D3QN_q_target_{}.pth'.format(episode))
        print("Loading target_net success!")

    def learning(self,tgr_update=False): # with a batch
        if not self.memory.warming_up():# 檢查memory是否有足夠的transition進行學習
            return
        
        if self.prioritized:
            tree_idx ,obs, info, actions, rewards, nx_obs, nx_info ,dones, ISweight = self.memory.sample_buffer()
            Isweight_tensor = torch.tensor(ISweight).to(device)
        else :
            obs, info, actions, rewards, nx_obs, nx_info ,dones = self.memory.sample_buffer()

        idx = torch.arange(self.batch_size, dtype=torch.long).to(device)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        info_tensor = torch.tensor(info,dtype=torch.float).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
        nx_obs_tensor = torch.tensor(nx_obs, dtype=torch.float32).to(device)
        nx_info_tensor = torch.tensor(nx_info,dtype=torch.float).to(device)
        dones_tensor = torch.tensor(dones).to(device)

        with torch.no_grad():
            # Double DQN
            q_ = self.target_net.forward(nx_obs_tensor,nx_info_tensor,dim=1)# target net 進行估計target value
            max_actions = torch.argmax(self.eval_net.forward(nx_obs_tensor,nx_info_tensor,dim=1), dim=-1)# eval net找出最大Q值action
            target = rewards_tensor + self.gamma * q_[idx, max_actions] #計算出target Q value
        
        q = self.eval_net.forward(obs_tensor, info_tensor,dim=1)[idx, actions_tensor]# q value

        if self.prioritized:
            abs_errors = (torch.abs(q - target).data).cpu().numpy() #計算TD error

            self.memory.batch_update(tree_idx, abs_errors) #更新priority

            loss = (Isweight_tensor * F.mse_loss(q, target.detach())).mean()
        else:
            loss = F.mse_loss(q, target.detach())# calculate loss (q-q')^2

        self.eval_net.optimizer.zero_grad()# 清除之前所求得的grad
        loss.backward()# 求grad
        self.eval_net.optimizer.step()# 更新parameters
        if not self.noisyNet:
            self.reduce_epsilon() #減小epsilon

        if tgr_update:
            self.soft_update() # soft update target net
            # self.hard_update() # hard update target net
    
    def get_epsilon(self):
        return self.epsilon