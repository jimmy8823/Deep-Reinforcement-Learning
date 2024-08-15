from stable_baselines3.common.env_checker import check_env
from cmvae_env import ueEnv as cmvae_env
from cnn_env import ueEnv as cnn_env
import datetime
import torch.nn as nn
from stable_baselines3 import PPO
import os
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
import torch as T
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import time

def test_ppo_with_CMVAE():
    policy_kwargs = dict(activation_fn=nn.ReLU,
                        net_arch=dict(pi=[64, 64, 64], vf=[32, 32]))

    env = cmvae_env()

    model = PPO.load("D:\\CODE\\Python\\AirSim\\PPO\\PPO_CMVAE\\model\\quad_land_2_100000_steps")
    
    for i in range(150):
        obs, _ = env.reset_test()
        terminal_state = ""
        done = False
        success = False
        exceed = False
        start = time.time()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, done, truncated, terminated_info = env.step(action)
            collision = terminated_info["collision"]
            success = terminated_info["success"]
            exceed = terminated_info["exceed"]
            if done :
                if collision:
                    terminal_state = "Collision"
                elif exceed:
                    terminal_state = "Time exceed"
                elif success:
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

def test_ppo_with_CNN():

    env = cnn_env()

    model = PPO.load("D:\\CODE\\Python\\AirSim\\PPO\\PPO_CNN\\model\\quad_land_2_100000_steps")
    print(model.policy)
    for i in range(150):
        obs, dict = env.reset_test()
        terminal_state = ""
        done = False
        success = False
        exceed = False
        start = time.time()
        while True:
            action, _states = model.predict(obs,deterministic=False)
            obs, rewards, done, truncated, terminated_info = env.step(action)
            collision = terminated_info["collision"]
            success = terminated_info["success"]
            exceed = terminated_info["exceed"]
            if done :
                if collision:
                    terminal_state = "Collision"
                elif exceed:
                    terminal_state = "Time exceed"
                elif success:
                    terminal_state = "Success landing"
                
                end = time.time()
                stopWatch = end - start
                
                #print("episode:{0}, reward: {1:.2f}, total steps: {2}, loss: {3:.4f}, epsilon: {4:.2f}".format(self.episode, episode_reward, step, episode_loss, self.D3QN.epsilon), flush=True)
                print("Episode is done, episode time: ", stopWatch)
                print("Result : {}".format(terminal_state))
                print("-----------------------------------------------------", flush=True)
                """
                with open("test_result.txt","a")as f:
                    f.write("Episode time: {0:.2f}, Result: {1}\n".format(stopWatch,terminal_state))
                """
                break

def test():
    cnn = nn.Sequential(
                    nn.Conv2d(2, 16, kernel_size=8, stride=4, padding=0),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
    input = T.randn(1,2,144,256)
    output = cnn(input)
    print(output.shape)

test_ppo_with_CNN()