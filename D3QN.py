import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_dim, memory_size, batch_size):
        self.size = memory_size
        self.batch_size = batch_size
        self.count = 0
        
        self.state_memory = np.zeros((self.size, state_dim))
        self.action_memory = np.zeros((self.size, ))
        self.reward_memory = np.zeros((self.size, ))
        self.next_state_memory = np.zeros((self.size, state_dim))
        self.terminal_memory = np.zeros((self.size, ), dtype=np.bool_)

    def store_transition(self, state, action, reward, nx_state, done):
        # prevent index(count) exceed memory size
        idx = self.count % self.size
        
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = nx_state
        self.terminal_memory[idx] = done

        self.count+=1
    def sample_buffer(self):
        # random sample a batch of transition from memory replay
        real_volume = min(self.size,self.count)
        sample = np.random.choice(real_volume, self.batch_size, replace=False)
        states = self.state_memory[sample]
        actions = self.action_memory[sample]
        rewards = self.reward_memory[sample]
        nx_states = self.next_state_memory[sample]
        dones = self.terminal_memory[sample]

        return states, actions, rewards, nx_states, dones
    
    def warning_up(self):
        return self.count > self.batch_size

#input_images_ch = 2
#batch_size = 16
    # input_images (batch,channel,256,192)
class Net(nn.Module):
    def __init__(self,lr,state_dim,action_dim):
        super().__init__()
        # conv 1
        self.conv1 = nn.Conv2d(state_dim.shape[1],16,kernel_size=5,stride=2)
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
        self.fc2 = nn.Linear(128,64)

        # value function & advantage function(Dueling)
        self.V = nn.Linear(64,1)
        self.A = nn.Linear(64,action_dim)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)

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
        V = self.V(out)
        A = self.A(out)
        Q = V + A - torch.mean(A,dim=-1,keepdim=True)
        return Q
        
    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

class D3QN():
    def __init__(self, lr, state_dim, action_dim, gamma=0.99, tau=0.005,
                 epsilon=1.0, eps_end=0.01, eps_dec=5e-7, memory_size=5000,
                 batch_size=16, ckpt_dir=""):
        super().__init__()
        self.gamma = gamma #discount factor
        self.tau = tau #
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.checkpoint_dir = ckpt_dir # checkpoint save 
        self.batch_size = batch_size
        self.action_space = action_dim

        self.eval_net = Net(lr,state_dim,action_dim)
        self.target_net = Net(lr,state_dim,action_dim)
        self.memory = ReplayBuffer(state_dim, memory_size, batch_size)

        self.soft_update(tau=1.0)

    def soft_update(self, tau=None):#軟更新，在更新target時保持一定比率的
        if tau is None:
            tau = self.tau

        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)
        print("soft update")

    def hard_update(self): # 過幾回合後將eval網路權重複製給target網路
        self.target_net.load_state_dict(self.eval_net.state_dict())
        print("hard update")
    
    def store_transition(self, state, action, reward, nx_state, done):
        self.memory.store_transition(state, action, reward, nx_state, done)

    def choose_action(self, observation, isTrain=True):
        state = torch.tensor([observation], dtype=torch.float).to(device)
        q_values = self.eval_net.forward(state) # get each action`s Q value
        action = torch.argmax(q_values).item()# 加入item()才可以拿到index 否則會回傳如tensor(5)

        if(np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.action_space)
        return action

    def reduce_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self, episode):
        self.eval_net.save_checkpoint(self.checkpoint_dir + 'Q_eval/D3QN_q_eval_{}.pth'.format(episode))
        print("Saving eval_net success!")
        self.target_net.save_checkpoint(self.checkpoint_dir + 'Q_target/D3QN_q_target_{}.pth'.format(episode))
        print("Saving target_net success!")
    def load_models(self, episode):
        self.eval_net.load_checkpoint(self.checkpoint_dir + 'Q_eval/D3QN_q_eval_{}.pth'.format(episode))
        print("Loading eval_net success!")
        self.target_net.load_checkpoint(self.checkpoint_dir + 'Q_target/D3QN_q_target_{}.pth'.format(episode))
        print("Loading target_net success!")

    def learning(self): # with a batch
        if not self.memory.warning_up():# 檢查memory是否有足夠的transition進行學習
            return
        states, actions, rewards, nx_states, dones = self.memory.sample_buffer()
        idx = torch.arange(self.batch_size, dtype=torch.long).to(device)
        states_tensor = torch.tensor(states, dtype=torch.float).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(device)
        nx_states_tensor = torch.tensor(nx_states, dtype=torch.float).to(device)
        dones_tensor = torch.tensor(dones).to(device)

        with torch.no_grad():
            # Double DQN
            q_ = self.target_net.forward(nx_states_tensor)# target net 進行估計target value
            max_actions = torch.argmax(self.eval_net.forward(nx_states_tensor), dim=-1)# eval net找出最大Q值action
            target = rewards_tensor + self.gamma * q_[idx, max_actions] #計算出target Q value
        
        q = self.eval_net.forward(states)[idx, actions_tensor]# q value

        loss = F.mse_loss(q, target.detach())# calculate loss (q-q')^2
        self.eval_net.optimizer.zero_grad()# 清除之前所求得的grad
        loss.backward()# 求grad
        self.eval_net.optimizer.step()# 更新parameters
        self.reduce_epsilon() #減小epsilon

        self.soft_update() # soft update target net
        #self.hard_update() # hard update target net
