import airsim
import os
from D3QN import D3QN
from utils.util import create_dir, plot_learning_curve

def Uav_move(action):
    if action == 0 :
        movebyastnc
def main():
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    agent = D3QN(lr=0.001, state_dim=, action_dim=13, gamma=0.99, tau=0.005,
                 epsilon=1.0, eps_end=0.05, eps_dec=5e-7, memory_size=5000,
                 batch_size=16)
    create_dir('/')

if __name__ == '__main__':
    main()