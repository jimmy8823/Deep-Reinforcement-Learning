import torch
import torch.nn.functional as F
import numpy as np
from setuptools import glob
import os, math

class QNet(torch.nn.Module):
    def __init__(self, action_dim):
        super(QNet, self).__init__()
        self.CNN = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3,stride=4),
                        torch.nn.ReLU(True),
                        torch.nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=4),
                        torch.nn.ReLU(True),
                        torch.nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=2,padding=1),
                        torch.nn.ReLU(True),
                   )

        self.lstm = torch.nn.LSTM(1208, 320, batch_first=True)
        
        self.V = torch.nn.Sequential(
                    torch.nn.Linear(320, 80),
                    torch.nn.ReLU(True),
                    torch.nn.Linear(80, 1)
                )
        self.A = torch.nn.Sequential(
                    torch.nn.Linear(320, 80),
                    torch.nn.ReLU(True),
                    torch.nn.Linear(80, action_dim)
                )
        
    def forward(self, x, depth, h, c):
        x = self.CNN(x) 
        x = x.view(x.size(0), -1).unsqueeze(1)
        depth = depth.view(depth.size(0), -1).unsqueeze(1)
        x = torch.cat((x, depth), dim = 2) # 1 1 1208

        x, (new_h, new_c) = self.lstm(x, (h, c))

        A = self.A(x)
        V = self.V(x)
        Q = V + A - A.mean(2).view(-1, 1, 1)

        return Q, new_h, new_c


class D3RQN:
    def __init__(self, action_dim, learning_rate, gamma, epsilon, target_update, steps):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.q_net = QNet(action_dim).to(self.device)
        self.target_q_net = QNet(action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = learning_rate)
        
        self.count = 0
        self.steps_done = steps
        self.action_dim = action_dim 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.eps_start = 0.999
        self.eps_end = 0.001
        self.eps_decay = 100000
        self.eps_steps = steps
        self.validaton = False

        self.target_update = target_update

    def choose_action(self, state_rgb, ac_state, h, c):
        with torch.no_grad():
            q_value, h, c = self.q_net(state_rgb, ac_state, h, c)

        if self.validaton:
            action = q_value.cpu().detach().argmax().item()
            return action
        
        self.epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.eps_steps / self.eps_decay)
        self.steps_done += 1
        self.eps_steps += 1

        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            action = q_value.cpu().detach().argmax().item()

        return action, h, c

    def update(self, loss):        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            print("[+] Update from target network")
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


    def init_hidden_state(self, batch_size, training=None):
        assert training is not None, "training step parameter should be dtermined"
        
        if training is True:
            return torch.zeros([1, batch_size, 320]).to(self.device), torch.zeros([1, batch_size, 320]).to(self.device)
        else:
            return torch.zeros([1, 1, 320]).to(self.device), torch.zeros([1, 1, 320]).to(self.device)
    
    def save_models(self, episode):
        checkpoint = {
            "net": self.q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done
        }
        torch.save(checkpoint, ".//model//D3RQN_{}.pth".format(episode))
    
    def load_models(self):
        files = glob.glob('.//model//*.pth')
        files.sort(key=os.path.getmtime)
        if len(files) > 0:
            print("[+] Loading D3RQN " + files[-1] + " ...")
            checkpoint = torch.load(files[-1])
            self.q_net.load_state_dict(checkpoint["net"])
            self.target_q_net.load_state_dict(checkpoint["net"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.steps_done = checkpoint["steps_done"]