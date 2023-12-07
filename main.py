import airsim
import os
from D3QN import D3QN
from utils.util import create_dir, plot_learning_curve
import numpy as np
import PIL.Image 
import time
import torch

client = airsim.MultirotorClient()

def get_observation():
    size_depth = 0
    size_rgb = 0
    while size_depth != 36864 or size_rgb != 110592: #防止有時候render失敗
        responses = client.simGetImages([
            airsim.ImageRequest("bottom_center", airsim.ImageType.Scene,False,False),
            airsim.ImageRequest("bottom_center", airsim.ImageType.DepthPlanar,True)])
        # stack images
        depth_image = np.array(responses[1].image_data_float, dtype=np.float64)
        rgb_image = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        size_depth = depth_image.size
        size_rgb = rgb_image.size

    depth_image = np.reshape(depth_image, (responses[1].height, responses[1].width)) 
    depth_image = np.array(PIL.Image.fromarray(depth_image).convert("L"))

    rgb_image = rgb_image.reshape(responses[0].height, responses[0].width, 3)
    gray_image = np.array(PIL.Image.fromarray(rgb_image).convert("L"))
    # 處理深度圖
    # 處理灰度影像
    obs = np.stack([depth_image, gray_image],dtype=np.float32, axis=-1)
    obs = np.expand_dims(obs, axis=0)

    obs = obs.transpose(0,3,1,2)

    # check collision & get attd、velocity
    kinematics = client.getMultirotorState()
    Collision_info = client.simGetCollisionInfo()
    velocity = kinematics.kinematics_estimated.linear_velocity.to_numpy_array()
    angular_val = kinematics.kinematics_estimated.angular_velocity.to_numpy_array()
    uav_pose = client.simGetVehiclePose()
    pose_attd = uav_pose.position.z_val
    info = np.concatenate((velocity,angular_val),axis=0)
    info = np.append(info,pose_attd)
    info = np.float32(info)

    # return state(images, info) info(vel,ang,attd) collision info
    return obs, info, Collision_info

def cal_reward(obs, nx_obs, action, info, collision, success, reverse, last_action):
    reward = 0 # step reward
    if collision :
        reward += -1 # collision reward
    if success :
        reward += 1 # success reward
    if action == 12: 
        reward += 0.05 # encourage agent descend
    if last_action == reverse:
        reward += -0.05 # prevent agent do reverse action repeatly
    # print("get reward : ", reward)
    obs = obs[0][0]
    nx_obs = nx_obs[0][0]
    depth_reward = -(np.sum(obs - nx_obs)/(256*144))/50
    if action !=12:
        reward += depth_reward
    return reward

def step(obs,action,last_action,step):
    # do the action
    reverse = -2
    done = False
    if action == 0 : # 0/1 forward
        client.moveByVelocityBodyFrameAsync(vx=0.5, vy=0, vz=0, duration=0.2)
        reverse = 2
    elif action == 1 :
        client.moveByVelocityBodyFrameAsync(vx=1, vy=0, vz=0, duration=0.2)
        reverse = 3
    elif action == 2 : # 2/3 backward
        client.moveByVelocityBodyFrameAsync(vx=-0.5, vy=0, vz=0, duration=0.2)
        reverse = 0
    elif action == 3 :
        client.moveByVelocityBodyFrameAsync(vx=-1, vy=0, vz=0, duration=0.2)
        reverse = 1
    elif action == 4 : # turn right 30 with speed 1m/s
        client.moveByVelocityBodyFrameAsync(1,0,0,0.2,yaw_mode={'is_rate': True, 'yaw_or_rate': 30})
        reverse = 8
    elif action == 5 : # turn right 50 with speed 1m/s
        client.moveByVelocityBodyFrameAsync(1,0,0,0.2,yaw_mode={'is_rate': True, 'yaw_or_rate': 50})
        reverse = 9
    elif action == 6 : # turn right 30 with speed 0.5m/s
        client.moveByVelocityBodyFrameAsync(0.5,0,0,0.2,yaw_mode={'is_rate': True, 'yaw_or_rate': 30})
        reverse = 10
    elif action == 7 : # turn right 50 with speed 0.5m/s
        client.moveByVelocityBodyFrameAsync(0.5,0,0,0.2,yaw_mode={'is_rate': True, 'yaw_or_rate': 50})
        reverse = 11
    elif action == 8 : # turn left 30 with speed 1 m/s
        client.moveByVelocityBodyFrameAsync(1,0,0,0.2,yaw_mode={'is_rate': True, 'yaw_or_rate': -30})
        reverse = 4
    elif action == 9 : # turn left 50 with speed 1 m/s
        client.moveByVelocityBodyFrameAsync(1,0,0,0.2,yaw_mode={'is_rate': True, 'yaw_or_rate': -50})
        reverse = 5
    elif action == 10 : # turn left 30 with speed 0.5 m/s
        client.moveByVelocityBodyFrameAsync(0.5,0,0,0.2,yaw_mode={'is_rate': True, 'yaw_or_rate': -30})
        reverse = 6
    elif action == 11 : # turn left 50 with speed 0.5 m/s
        client.moveByVelocityBodyFrameAsync(0.5,0,0,0.2,yaw_mode={'is_rate': True, 'yaw_or_rate': -50})
        reverse = 7
    elif action == 12 : # descend
        client.moveByVelocityBodyFrameAsync(0,0,1,0.2)
        
    nx_obs, nx_info, collision_info = get_observation()
    success = False
    collision = collision_info.has_collided
    if collision or step >= 2000: # collision or exceed maximum step terminal this episode
        done = True
    
    if nx_info[6] > -1: # when the UAV attd less than 1m trigger landing procedure
        collision = True
        client.landAsync().join()
        nx_obs, nx_info, collision_info = get_observation()
        if collision_info.object_name.startswith("Ground"):
            success = True
            collision = False
        done = True
        
    reward = cal_reward(obs,nx_obs, action, nx_info, collision_info.has_collided,
                        success,reverse,last_action)
    
    # return next_state(images, info), reward, done
    return nx_obs, nx_info, reward, done, collision

def main():
    # state [batch, channel , height, width]
    eps_list =[]
    reward_list =[]
    step_list =[]
    collision_list=[]
    state_dim = [16,2,144,256]
    episodes = 3000
    # connect to the AirSim simulator
    agent = D3QN(lr=0.001, obs_dim=state_dim, info_dim=7, action_dim=13, gamma=0.99, tau=0.005,
                 epsilon=1.0, eps_end=0.05, eps_dec=5e-7, memory_size=5000,
                 batch_size=16,ckpt_dir="./checkpoint")
    # tau for softupdate , gamma for discount factor , lr for learning rate
    # create_dir('/')
    for episode in range(episodes): # training
        eps = episode+1
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync()
        last_action = -1
        total_reward = 0
        step_count = 0
        done = False
        collision = False
        position = airsim.Vector3r(20,0,-50)
        Pose = airsim.Pose(position)
        client.simSetVehiclePose(Pose,ignore_collision=False,vehicle_name="")
        time.sleep(3)
        # client.confirmConnection()
        obs, info, collision_info = get_observation()
        #obs (1, 2, 144, 256)
        while not done :
            step_count +=1
            action = agent.choose_action(obs,info,isTrain=True)
            nx_obs, nx_info, reward, done, col = step(obs,action,last_action,step_count)# do action and get next obs, reward, info
            last_action = action
            agent.store_transition(obs,info,action,reward,nx_obs,nx_info,done)
            agent.learning()
            collision = col
            total_reward += reward
            obs = nx_obs
            info = nx_info
        print("Episode :{0}, avg_step: {3}, reward {1:.3f}, current epsilon :{2:.3f} ,Collision : {4}".format(eps,total_reward,agent.get_epsilon(),step_count,collision))
        eps_list.append(eps)
        reward_list.append(total_reward)
        step_list.append(step_count)
        collision_list.append(collision)
        if (eps) % 100 == 0:
            agent.save_models(eps)
    # record learning result
    with open("./result/leanring_result.txt", "w") as fp:
        for i in len(eps_list):
            fp.write("%s : %.3f, %d, %d \n" % eps_list[i],reward_list[i],step_list[i],collision_list[i])
    
    plot_learning_curve(eps_list, reward_list, title='Reward', ylabel='reward',figur_file="./result/reward")
    plot_learning_curve(eps_list, step_list, title='step', ylabel='avg_step',figur_file="./result/step")

    
if __name__ == '__main__':
    main()