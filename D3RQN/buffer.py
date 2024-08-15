

import torch
from os import error
import numpy as np
import random

# N-Step Memory
class NStepMemory(object):
    # memory buffer to store episodic memory
    def __init__(self, memory_size=150, n_steps=4, gamma = 0.99):
        self.buffer = []
        self.memory_size = memory_size
        self.n_steps = n_steps
        self.gamma = gamma
        self.next_idx = 0
        
    def push(self, state_rgb, ac_state, action, reward, next_state_rgb, next_ac_state, done, h, c, next_h, next_c):
        # Tensor to numpy
        state_rgb = state_rgb.detach().cpu().numpy()
        next_state_rgb = next_state_rgb.detach().cpu().numpy()
        ac_state = ac_state.detach().cpu().numpy()
        next_ac_state = next_ac_state.detach().cpu().numpy()
        h = h.detach().cpu().numpy()
        c = c.detach().cpu().numpy()
        next_h = next_h.detach().cpu().numpy()
        next_c = next_c.detach().cpu().numpy()

        data = (state_rgb, ac_state, action, reward, next_state_rgb, next_ac_state, done, h, c, next_h, next_c)
        if len(self.buffer) <= self.memory_size:    # buffer not full
            self.buffer.append(data)
        else:                                       # buffer is full
            self.buffer[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def get(self, i):
        # sample episodic memory
        # [state_rgb, ac_state, action, reward, next_state_rgb, next_ac_state, done, h, c, next_h, next_c]

        begin = i
        finish = i + self.n_steps if(i + self.n_steps < self.size()) else self.size()
        sum_reward = 0 # n_step rewards
        data = self.buffer[begin:finish]
        state_rgb = data[0][0]
        ac_state = data[0][1]    
        action = data[0][2]
        h = data[0][7]
        c = data[0][8]
        for j in range(len(data)):
            # compute the n-th reward
            sum_reward += (self.gamma**j) * data[j][3]
            if data[j][6]:
                # manage end of episode
                next_state_rgb = data[j][4]
                next_ac_state = data[j][5]
                next_h = data[j][9]
                next_c = data[j][10]
                done = 1
                break
            else:
                next_state_rgb = data[j][4]
                next_ac_state = data[j][5]
                next_h = data[j][9]
                next_c = data[j][10]
                done = 0

        return state_rgb, ac_state, action, sum_reward, next_state_rgb, next_ac_state, done, h, c, next_h, next_c
    
    def reset(self):
        self.buffer = []
        self.next_idx = 0

    def size(self):
        return len(self.buffer)


# Prioritized Experience Replay
class Buffer(object):
    def __init__(self, memory_size = 4000, burn_in = 4, a = 0.6, e = 0.01):
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.burn_in_len = burn_in
        self.prio_max = 0.1
        self.a = a
        self.e = e
    
    def push(self, data):
        data = (data)
        p = (np.abs(self.prio_max) + self.e) ** self.a
        self.tree.add(p, data)

    def sample(self, batch_size):
        state_rgbs, ac_states, actions, rewards, next_state_rgbs, next_ac_states, dones, hs, cs, next_hs, next_cs = [], [], [], [], [], [], [], [], [], [], []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        burn_in_list = np.random.choice(batch_size, size=self.burn_in_len, replace=False)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            state_rgb, ac_state, action, reward, next_state_rgb, next_ac_state, done, h, c, next_h, next_c = data
            state_rgbs.append(state_rgb)
            ac_states.append(ac_state)
            actions.append(action)
            rewards.append(reward)
            next_state_rgbs.append(next_state_rgb)
            next_ac_states.append(next_ac_state)
            dones.append(done)
            if i in burn_in_list:
                hs.append(h)
                cs.append(c)
                next_hs.append(next_h)
                next_cs.append(next_c)
            else:
                hs.append(np.zeros((1, 1, 320), dtype=np.float32))
                cs.append(np.zeros((1, 1, 320), dtype=np.float32))
                next_hs.append(np.zeros((1, 1, 320), dtype=np.float32))
                next_cs.append(np.zeros((1, 1, 320), dtype=np.float32))
            priorities.append(p)
            idxs.append(idx)
        
        return idxs, state_rgbs, ac_states, actions, rewards, next_state_rgbs, next_ac_states, dones, hs, cs, next_hs, next_cs
    
    def update(self, idxs, errors):
        self.prio_max = max(self.prio_max, np.max(np.abs(errors)))
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.e) ** self.a
            self.tree.update(idx, p)
    
    def size(self):
        return self.tree.n_entries

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    #update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    #find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    #store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    #update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)
    
    #get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

    
