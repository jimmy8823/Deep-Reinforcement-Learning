# Python client example to get Lidar data from a drone
#

import airsim

import sys
import math
import time
import argparse
import pprint
import numpy

# Makes the drone fly and get Lidar data
class LidarTest:

    def __init__(self):

        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

    def execute(self):

        print("arming the drone...")
        self.client.armDisarm(True)

        state = self.client.getMultirotorState()
        s = pprint.pformat(state)
        #print("state: %s" % s)

        airsim.wait_key('Press any key to takeoff')
        self.client.takeoffAsync().join()

        state = self.client.getMultirotorState()
        #print("state: %s" % pprint.pformat(state))

        airsim.wait_key('Press any key to move vehicle to (-10, 10, -2) at 5 m/s')
        self.client.moveToPositionAsync(-10, 10, -2, 5).join()

        self.client.hoverAsync().join()

        airsim.wait_key('Press any key to get Lidar readings')
        
        for i in range(1,5):
            distance_front = self.client.getDistanceSensorData(distance_sensor_name="DistanceFront", vehicle_name="Drone1")
            print(distance_front)
    """     distance_front = self.client.getDistanceSensorData(distance_sensor_name="DistanceFront", vehicle_name="Drone1")
            print(f"Front Distance: {distance_front.distance} meters")

            # 獲取後方距離傳感器數據
            distance_back = self.client.getDistanceSensorData(distance_sensor_name="DistanceBack", vehicle_name="Drone1")
            print(f"Back Distance: {distance_back.distance} meters")

            # 獲取左方距離傳感器數據
            distance_left = self.client.getDistanceSensorData(distance_sensor_name="DistanceLeft", vehicle_name="Drone1")
            print(f"Left Distance: {distance_left.distance} meters")

            # 獲取右方距離傳感器數據
            distance_right = self.client.getDistanceSensorData(distance_sensor_name="DistanceRight", vehicle_name="Drone1")
            print(f"Right Distance: {distance_right.distance} meters")
            time.sleep(5)"""

    def parse_lidarData(self, data):

        # reshape array of floats to array of [X,Y,Z]
        points = numpy.array(data.point_cloud, dtype=numpy.dtype('f4'))
        points = numpy.reshape(points, (int(points.shape[0]/3), 3))
       
        return points

    def write_lidarData_to_disk(self, points):
        # TODO
        print("not yet implemented")

    def stop(self):

        airsim.wait_key('Press any key to reset to original state')

        self.client.armDisarm(False)
        self.client.reset()

        self.client.enableApiControl(False)
        print("Done!\n")

# main
if __name__ == "__main__":
    args = sys.argv
    args.pop(0)

    arg_parser = argparse.ArgumentParser("Lidar.py makes drone fly and gets Lidar data")

    arg_parser.add_argument('-save-to-disk', type=bool, help="save Lidar data to disk", default=False)
  
    args = arg_parser.parse_args(args)    
    lidarTest = LidarTest()
    try:
        lidarTest.execute()
    finally:
        lidarTest.stop()