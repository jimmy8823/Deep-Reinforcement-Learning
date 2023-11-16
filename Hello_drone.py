# ready to run example: PythonClient/multirotor/hello_drone.py
import airsim
import os

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Async methods returns Future. Call join() to wait for task to complete.
client.takeoffAsync().join()
client.moveToPositionAsync(10 ,50 ,-50 ,5).join()
print("moving")
#client.moveByVelocityAsync(5, 0, -3, 10).join() # 第三阶段：以1m/s速度向前飞5秒钟
#client.moveByVelocityAsync(0, 5, -3, 5).join() # 第三阶段：以1m/s速度向右飞5秒钟
#client.moveByVelocityAsync(-5, 0, -3, 5).join() # 第三阶段：以1m/s速度向后飞5秒钟
#client.moveByVelocityAsync(0, -5, -3, 5).join() # 第三阶段：以1m/s速度向左飞5秒钟 作者：皮卡丘上大学啦 https://www.bilibili.com/read/cv23845978?from=articleDetail 出处：bilibili

# take images
responses = client.simGetImages([
    airsim.ImageRequest("bottom_center", airsim.ImageType.Scene,True),
    airsim.ImageRequest("bottom_center", airsim.ImageType.DepthVis,True)])
print('Retrieved images: %d', len(responses))

# do something with the images

for idx ,response in enumerate(responses):
    print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
    print(response.image_data_float)
    # airsim.write_file('py%d.png'%idx, response.image_data_uint8)

    """if response.pixels_as_float:
        print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
        airsim.write_pfm('py%d.pfm'%idx, airsim.get_pfm_array(response))
    else:
    """
        