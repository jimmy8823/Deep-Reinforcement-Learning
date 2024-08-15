## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2


def cut(depth_image, gray_image):
    depth_image = cv2.resize(depth_image[10:470,30:610],(256,144),interpolation=cv2.INTER_CUBIC)
    gray_image = cv2.resize(gray_image[10:470,30:610],(256,144),interpolation=cv2.INTER_CUBIC)
    return depth_image, gray_image
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

hole_filling_filter = rs.hole_filling_filter()
hole_filling_mode = 1  # 0, 1, or 2
hole_filling_filter.set_option(rs.option.holes_fill, hole_filling_mode)

align_to = rs.stream.color
align = rs.align(align_to)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        filtered_depth_frame = hole_filling_filter.process(depth_frame)
        depth_image = np.asanyarray(filtered_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        #depth_image/=1000
        
        depth_image , color_image = cut(depth_image, color_image)
        #print(depth_image)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape
        max_depth = np.max(depth_image.flatten())
        min_depth = np.min(depth_image.flatten())
        print(max_depth, min_depth)
        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()