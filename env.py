import airsim
import math
import numpy as np
import time
from PIL import Image
import random

class ueEnv(object):
    def __init__(self):
        self.D3RQN = False
        self.client = airsim.MultirotorClient()
        self.map_index = 3
        self.map = ["Level_one", "Level_two", "Level_one_without_foliage", "Level_two_without_foliage"]
        self.start_pos = [(9.2,-33.6,-25), (-2,42.4,-25), (-44.1,38,-25), 
                 (-44.1,-25.4,-25), (0,0,-25),(20,30,-25),
                 (-51,1,-25),(-38,-1,-25),(-20,45,-25),
                 (-38,-22,-25),(10.5, -6, -25),(15.9, 42, -25),
                 (-12.8, 20,-25),(-27.3, 44.7,-25),(-4, -9.7, -25)]
        self.episode_count = 0
        #self.client.simLoadLevel(self.map[self.map_index])
        self.client.simSetSegmentationObjectID("[\w]*", 255, True)
        self.client.simSetSegmentationObjectID("SM_unsafe[\w]*", 0, True)
        
        # Variable For test
        self.test_pos = [(43,38,-25) ,(-12.8,30,-25), (-39,37.1,-28),
                         (-42.3,-44,-25), (-14.8,-52,-25), (35.5,-43.8,-25),
                         (65,-43.1,-25), (46.2,1.5,-25), (24.6,-23.4,-25),
                         (4.7,7.3,-25),(41.6,27.1,-5)]
        self.orientation_count = 0
        self.pos_count = 0

    def get_obs(self):
        """
        return shape[1,2,144,256]
        """
        size_depth = 0
        size_rgb = 0
        while size_depth != 36864 or size_rgb != 110592: #防止有時候render失敗
            responses = self.client.simGetImages([
                airsim.ImageRequest("bottom_center", airsim.ImageType.Scene,False,False),
                airsim.ImageRequest("bottom_center", airsim.ImageType.DepthPlanar,True),
                airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation,False,False)
                ])
            # stack images
            rgb_image = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            depth_image = np.array(responses[1].image_data_float, dtype=np.float64)
            seg_image = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8)
            size_depth = depth_image.size
            size_rgb = rgb_image.size

        depth_image = np.reshape(depth_image, (responses[1].height, responses[1].width)) 
        depth_image = np.array(Image.fromarray(depth_image).convert("L"))
        depth_image[depth_image>25] = 25
        depth_image= depth_image/25 #noramlize
        
        rgb_image = rgb_image.reshape(responses[0].height, responses[0].width, 3)
        gray_image = np.array(Image.fromarray(rgb_image).convert("L"))
        gray_image= gray_image/255

        seg_image = seg_image.reshape(responses[2].height, responses[2].width, 3)
        bin_seg_image = np.array(Image.fromarray(seg_image).convert("L"))
        bin_seg_image = bin_seg_image/255


        obs = np.stack([gray_image, depth_image],dtype=np.float32, axis=-1)
        obs = np.expand_dims(obs, axis=0)

        obs = obs.transpose(0,3,1,2)

        # return state(images, info) info(vel,ang,attd) collision info
        return obs, bin_seg_image

    def get_info(self):
        # check collision & get attd、velocity
        kinematics = self.client.getMultirotorState()
        velocity = kinematics.kinematics_estimated.linear_velocity.to_numpy_array()
        angular_val = kinematics.kinematics_estimated.angular_velocity.to_numpy_array()
        uav_pose = self.client.simGetVehiclePose()
        pose_attd = uav_pose.position.z_val * -1
        normalize_attd = pose_attd/25
        info = np.concatenate((velocity,angular_val),axis=0)
        info = np.append(info,normalize_attd)
        info = np.float32(info)
        info = np.expand_dims(info, axis=0)

        Collision_info = self.client.simGetCollisionInfo()

        return info, Collision_info, pose_attd
    
    def step(self, action, step):
        duration = 1
        done = False
        collision = False
        exceed = False

        if action == 0 : # 0/1 forward
            self.client.moveByVelocityBodyFrameAsync(vx=0.5, vy=0, vz=0, duration=duration , drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 0.0))
        elif action == 1 :
            self.client.moveByVelocityBodyFrameAsync(vx=1, vy=0, vz=0, duration=duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 0.0))
        elif action == 2 : # turn right 30 with speed 1m/s
            self.client.moveByVelocityBodyFrameAsync(1,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 30.0))
        elif action == 3 : # turn right 30 with speed 0.5m/s
            self.client.moveByVelocityBodyFrameAsync(0.5,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 30.0))
        elif action == 4 : # turn left 30 with speed 1 m/s
            self.client.moveByVelocityBodyFrameAsync(1,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, -30.0))
        elif action == 5 : # turn left 30 with speed 0.5 m/s
            self.client.moveByVelocityBodyFrameAsync(0.5,0,0,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, -30.0))
        elif action == 6 :
            self.client.moveByVelocityBodyFrameAsync(0,0,1,duration,drivetrain=airsim.DrivetrainType.ForwardOnly, yaw_mode=airsim.YawMode(True, 0.0))
        time.sleep(0.2)

        nx_obs, nx_seg = self.get_obs()
        nx_info, collision_info, attd = self.get_info()
        success = False
        collision = collision_info.has_collided
        if collision : # collision  
            done = True

        if step >= 300: # exceed maximum step terminal this episode
            exceed = True
            done = True

        if attd < 1.5: # when the UAV attd less than 1m trigger landing procedure
            collision = True
            self.client.landAsync()
            while(not collision_info.has_collided):
                nx_info, collision_info, attd = self.get_info()
            if  collision_info.object_name.startswith("Ground") or \
                collision_info.object_name.startswith("ground") or \
                collision_info.object_name.startswith("Landscape"):
                done = True
                success = True
                collision = False
            else :
                done = True

        reward = self.cal_reward(nx_seg, action, nx_info, collision, success)

        return nx_obs, nx_info, reward, done, collision, exceed, success

    def cal_reward(self, seg, action, info, collision, success): # let reward range [1,-1]
        reward = 0 # step reward
        if collision :
            reward += -20 #collision reward
            return reward
        if success :
            reward += 50 #success reward
            return reward
        reward -= 0.2
        seg = seg[35:107,63:191]
        seg_reward = (np.mean(seg)-0.5)
        
        if seg_reward == 0.5: # means UAV on safe area
            if action != 6:
                seg_reward = 0
        reward += seg_reward

        if action == 6:
            reward += 1
        else:
            reward -= 0.05
        #print("get seg reward : ", seg_reward)
        
        return reward

    def reset(self):
        """
        Return
            np obs  [1,2,144,256]
            np info [7]
        """
        x,y,z = 0,0,0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync()
        self.episode_count += 1
        degree = random.randint(0, 360)
        orientation = airsim.to_quaternion(0, 0, math.radians(degree))
        if (self.map_index % 2) == 0 : # level one
            x = random.randint(-45, 45)
            y = random.randint(-45, 45)
            z = -25
        else :
            x,y,z = self.start_pos[self.episode_count%15]
        position = airsim.Vector3r(x,y,-25)
        pose = airsim.Pose(position,orientation)
        self.client.simSetVehiclePose(pose,True)

        print("[+] Start position: [{0}, {1}, {2}, degree: {3}]".format(x, y, z, degree), flush=True)
        time.sleep(1)
        obs, bin_seg_image = self.get_obs()
        info, _ , attd= self.get_info()
        return obs, info

    def reset_test(self):
        """
        Return
            np obs  [1,2,144,256]
            np info [7]
        """
        x,y,z = 0,0,0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync()
        self.episode_count += 1
        degree = self.orientation_count*24
        orientation = airsim.to_quaternion(0, 0, math.radians(degree))
        x,y,z = self.test_pos[self.pos_count]

        position = airsim.Vector3r(x,y,-25)
        pose = airsim.Pose(position,orientation)
        self.client.simSetVehiclePose(pose,True)
        print("[+] Start position: [{0}, {1}, {2}, degree: {3}]".format(x, y, z, degree), flush=True)
        with open("test_result.txt","a")as f:
            f.write("[+] Start position: [{0}, {1}, {2}, degree: {3}]\n".format(x, y, z, degree))
        self.orientation_count +=1 
        if self.orientation_count == 15:# every position test 15 different orientation
            with open("test_result.txt","a")as f:
                f.write("-----------------------Postition {}---------------------------------".format(self.pos_count+1))
            self.pos_count += 1
            self.orientation_count = 0
        obs, bin_seg_image = self.get_obs()
        info, _ , attd= self.get_info()
        return obs, info

"""env = ueEnv()

obs, info  = env.reset()
print(obs.shape)
print(info.shape)

print("obs : ", obs)
print("info : ", info)"""