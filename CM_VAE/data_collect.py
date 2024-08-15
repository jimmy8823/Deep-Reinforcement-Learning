import torch
import torch.nn.functional as F
import torch.nn as nn
from CM_VAE_ver2 import CmVAE
import airsim
import random
import time
import numpy as np
import PIL.Image
import os
import math 

# controling UAV to collect image data in AirSim 

def respawn(map,index):
    spawn_pos = [(9.2,-33.6,-25), (-2,42.4,-25), (-44.1,37.1,-15), 
                 (-44.1,-25.4,-20), (-44.1,-25.4,-25), (0,0,-20),
                 (-44.1,37.1,-20), (0,0,-15), (-44.1,37.1,-25)]
    degree = random.randint(0,360)
    orientation = airsim.to_quaternion(0, 0, math.radians(degree))
    if map == "Level_two" or map == "Level_two_without_foliage":
        position = spawn_pos[index]
    else:
        x = random.randint(-45, 45)
        y = random.randint(-45, 45)
        position = (x , y, -25)
    return position , orientation

def move(client):
    action = random.randint(0,12)
    duration = 1 
    if action == 0 : # 0/1 forward
        client.moveByVelocityBodyFrameAsync(vx=0.5, vy=0, vz=0, duration=duration , drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 0.0))
    elif action == 1 :
        client.moveByVelocityBodyFrameAsync(vx=1, vy=0, vz=0, duration=duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 0.0))
    elif action == 2 : # 2/3 backward
        client.moveByVelocityBodyFrameAsync(vx=-0.5, vy=0, vz=0, duration=duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 0.0))
    elif action == 3 :
        client.moveByVelocityBodyFrameAsync(vx=-1, vy=0, vz=0, duration=duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 0.0))
    elif action == 4 : # turn right 30 with speed 1m/s
        client.moveByVelocityBodyFrameAsync(1,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 30.0))
    elif action == 5 : # turn right 50 with speed 1m/s
        client.moveByVelocityBodyFrameAsync(1,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 50.0))
    elif action == 6 : # turn right 30 with speed 0.5m/s
        client.moveByVelocityBodyFrameAsync(0.5,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 30.0))
    elif action == 7 : # turn right 50 with speed 0.5m/s
        client.moveByVelocityBodyFrameAsync(0.5,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 50.0))
    elif action == 8 : # turn left 30 with speed 1 m/s
        client.moveByVelocityBodyFrameAsync(1,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, -30.0))
    elif action == 9 : # turn left 50 with speed 1 m/s
        client.moveByVelocityBodyFrameAsync(1,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, -50.0))
    elif action == 10 : # turn left 30 with speed 0.5 m/s
        client.moveByVelocityBodyFrameAsync(0.5,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, -30.0))
    elif action == 11 : # turn left 50 with speed 0.5 m/s
        client.moveByVelocityBodyFrameAsync(0.5,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, -50.0))
    elif action == 12 : # descend
        client.moveByVelocityBodyFrameAsync(0,0,1,duration)

        time.sleep(0.5)

def main():
    steps = 25000
    client = airsim.MultirotorClient()
    time.sleep(2)
    client.confirmConnection()
    # setting segmentation white pixel as ground ; black pixel as obstacle
    client.simSetSegmentationObjectID("[\w]*", 255, True)
    client.simSetSegmentationObjectID("SM_unsafe[\w]*", 0, True)
    index = 0
    for step in range(12500,steps):
        collision = client.simGetCollisionInfo()
        if step % 500 == 0 or collision.has_collided:
            coordinate , orientation = respawn("Level_two_without_foliage", index)
            client.reset()
            client.enableApiControl(True)
            client.armDisarm(True)
            client.takeoffAsync().join()
            position = airsim.Vector3r(coordinate[0],coordinate[1],coordinate[2])
            Pose = airsim.Pose(position,orientation)
            client.simSetVehiclePose(Pose,ignore_collision=True,vehicle_name="")
            index = (index + 1) % 9
            time.sleep(3)
        
        size_rgb = 0
        size_depth = 0
        while size_depth != 36864 or size_rgb != 110592: #防止有時候render失敗
            responses = client.simGetImages([
            airsim.ImageRequest("bottom_center", airsim.ImageType.Scene,False,False),
            airsim.ImageRequest("bottom_center", airsim.ImageType.DepthPlanar,True),
            airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation,False,False)])
            rgb_image = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            depth_image = np.array(responses[1].image_data_float, dtype=np.float32)
            seg_image = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
            size_depth = depth_image.size
            size_rgb = rgb_image.size

        depth_image = np.reshape(depth_image, (responses[1].height, responses[1].width)) 
        #depth_image = np.array(PIL.Image.fromarray(depth_image).convert("L"))
        rgb_image = rgb_image.reshape(responses[0].height, responses[0].width, 3)
        gray_image = np.array(PIL.Image.fromarray(rgb_image).convert("L"))
        seg_image = seg_image.reshape(responses[2].height, responses[2].width, 3)
        bin_seg_image = np.array(PIL.Image.fromarray(seg_image).convert("L"))
        move(client)
        
        dirname = os.path.dirname(__file__)
        depth = os.path.join(dirname, "training_data2/depth/depth_image_" + str(step+1) + ".pfm")
        gray = os.path.join(dirname, "training_data2/gray/gray_image_" + str(step+1) + ".png")
        seg = os.path.join(dirname, "training_data2/seg/bin_seg_image_" + str(step+1) + ".png")

        airsim.write_pfm(os.path.normpath(depth),depth_image)
        airsim.write_png(os.path.normpath(gray),gray_image)
        airsim.write_png(os.path.normpath(seg),bin_seg_image)

if __name__ == "__main__":
    main()

