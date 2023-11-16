import airsim
import os
from D3QN import D3QN
from utils.util import create_dir, plot_learning_curve

def get_observation(client):
    responses = client.simGetImages([
        airsim.ImageRequest("bottom_center", airsim.ImageType.Scene,True),
        airsim.ImageRequest("bottom_center", airsim.ImageType.DepthVis,True)])
    # stack images
    # return state(images, info)
    return 
def cal_reward(, is_collision):

def step(action):
    # do the action
    if action == 0 :
        moveByVelocityBodyFrameAsync(vx=, vy=, vz=, duration=0.2)
    elif action == 1 :
        moveByVelocityBodyFrameAsync()
    elif action == 2 :

    elif action == 3 :
    elif action == 4 :
    elif action == 5 :
    elif action == 6 :
    elif action == 7 :
    elif action == 8 :
    elif action == 9 :
    elif action == 10 :
    elif action == 11 :
    elif action == 12 :
    
    # return next_state(images, info), reward, done
    return 

def main():
    # state [batch, channel , height, width]
    state_dim = [16,2,256,192]
    episodes = 3000
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.reset()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    agent = D3QN(lr=0.001, state_dim=state_dim, action_dim=13, gamma=0.99, tau=0.005,
                 epsilon=1.0, eps_end=0.05, eps_dec=5e-7, memory_size=5000,
                 batch_size=16)
    # tau for softupdate , gamma for discount factor , lr for learning rate
    # create_dir('/')
    for epsidoe in range(episodes):
        total_reward = 0
        done = False
        observation = get_observation(client)

if __name__ == '__main__':
    main()