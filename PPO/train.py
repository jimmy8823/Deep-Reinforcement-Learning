from stable_baselines3.common.env_checker import check_env
from cmvae_env import ueEnv as cmvae_env
from cnn_env import ueEnv as cnn_env
import datetime
import torch.nn as nn
from stable_baselines3 import PPO
import os
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch as T
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=135)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        self.cnn = nn.Sequential(
                    nn.Conv2d(2, 16, kernel_size=8, stride=4, padding=0),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(3840,128)
                )

    def forward(self, observations) -> T.Tensor:
        feature = self.cnn(observations['image'])
        feature = T.cat((feature,observations['info']),dim=1)
        return feature

def create_dir():
    path = (datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
    model_path = "/model"
    os.mkdir(path)
    model_path = path + model_path
    os.mkdir(model_path)
    log_path = path + "/log"
    return model_path , log_path

def train_ppo_with_CMVAE():
    policy_kwargs = dict(activation_fn=nn.ReLU,
                        net_arch=dict(pi=[64, 64, 64], vf=[32, 32]))

    env = cmvae_env()

    #model_path, log_path = create_dir()
    model_path = "D:\\CODE\\Python\\AirSim\\PPO\\PPO_CMVAE\\model"
    log_path = "D:\\CODE\\Python\\AirSim\\PPO\\PPO_CMVAE\\log\\PPO_1"
    model = PPO.load("D:\\CODE\\Python\\AirSim\\PPO\\PPO_CMVAE\\model\\quad_land_150000_steps",env)
    """model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, \
                tensorboard_log=log_path, 
                gae_lambda = 0.95,
                gamma=0.99,  # lower 0.9 ~ 0.99
                n_steps=2000,
                vf_coef=0.5,
                ent_coef=0.001,
                max_grad_norm=0.5,
                batch_size =4,
                n_epochs =4,
                clip_range =0.2)
"""
    callbacks = [CheckpointCallback(
            save_freq=10000, save_path=model_path, name_prefix='quad_land_2'
            )]

    model.learn(total_timesteps=100000,progress_bar=True,callback=callbacks)

def train_ppo_with_cnn():
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[64, 64, 64], vf=[32, 32]),
        features_extractor_class=CustomCNN,
        share_features_extractor=True
    )
    
    env = cnn_env()

    #model_path, log_path = create_dir()
    model_path = "D:\\CODE\\Python\\AirSim\\PPO\\2024_06_22-13_25_39\\model"
    log_path = "D:\\CODE\\Python\\AirSim\\PPO\\2024_06_22-13_25_39\\log\\PPO_1"
    model = PPO.load("D:\\CODE\\Python\\AirSim\\PPO\\2024_06_22-13_25_39\\model\\quad_land_150000_steps",env)
    
    """model =  PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1, \
                tensorboard_log=log_path, 
                gae_lambda = 0.95,
                gamma=0.99,  # lower 0.9 ~ 0.99
                n_steps=2000,
                vf_coef=0.5,
                ent_coef=0.001,
                max_grad_norm=0.5,
                batch_size =4,
                n_epochs =4,
                clip_range =0.2)
    """
    print(model.policy)
    callbacks = [CheckpointCallback(
            save_freq=10000, save_path=model_path, name_prefix='quad_land_2'
            )]

    model.learn(total_timesteps=100000,progress_bar=True,callback=callbacks,reset_num_timesteps=False)

train_ppo_with_cnn()

"""
class CustomCombinedExtractor(BaseFeaturesExtractor):

    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.


    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(
                    nn.Conv2d(2, 16, kernel_size=8, stride=4, padding=0),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Flatten(),
                    nn.Linear(3840,128)
                )

                total_concat_size += 3840
            elif key == "info":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 7)
                total_concat_size += 7

        self.extractors = nn.ModuleDict(extractors)
        self.fc = nn.Linear(total_concat_size,128)
        # Update the features dim manually
        self._features_dim = 135

    def forward(self, observations) -> T.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            #print(key)
            #print(extractor(observations[key]))
            encoded_tensor_list.append(extractor(observations[key]))
        encode_tensor = T.cat(encoded_tensor_list, dim=1)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return encode_tensor

"""