import cv2
import numpy as np
import os
import inout

np.set_printoptions(precision=8, suppress=True)

# Dataset Info
######################################
# Path to the ROBI dataset
data_path = '/Add/your/path/here/ROBI/'
#data_path = 'D:/ROBI/'
obj = 'DSub_connector' # Object Name
# Camera sensor to use
sensor = 'Ensenso' #'RealSense'
# Scene ID
scene_ids = [1,2,3,4,5,6,7,8,9]
######################################



# Camera Info
#######################################################
cam_info_path = os.path.join('./cam', '{}.yml')
cam_DEPTH_path = cam_info_path.format(sensor+'_DEPTH')
cam_LEFT_path = cam_info_path.format(sensor+'_LEFT')
cam_RIGHT_path = cam_info_path.format(sensor+'_RIGHT')
cam_RGB_path = cam_info_path.format(sensor+'_RGB')

depth_info = inout.load_cam_info(cam_DEPTH_path)
left_info = inout.load_cam_info(cam_LEFT_path)
right_info = inout.load_cam_info(cam_RIGHT_path)
RGB_info = inout.load_cam_info(cam_RGB_path)

disparity_shift = depth_info['disparity_shift']
baseline = depth_info['baseline']
depth_unit = depth_info['depth_unit']
depth_fx = depth_info['fx']
depth_fy = depth_info['fy']
depth_cx = depth_info['cx']
depth_cy = depth_info['cy']
left_fx = left_info['fx']
left_fy = left_info['fy']
left_cx = left_info['cx']
left_cy = left_info['cy']
right_fx = right_info['fx']
right_fy = right_info['fy']
right_cx = right_info['cx']
right_cy = right_info['cy']
#######################################################



# Data path Info
########################################################################################################
# For all cameras
scene_list_path = os.path.join(data_path, obj, 'Scene_{}','{}','list.txt')
scene_DEPTH_path = os.path.join(data_path, obj, 'Scene_{}','{}','Depth','DEPTH_{}.png')
scene_LEFT_PATTERN_path = os.path.join(data_path, obj, 'Scene_{}','{}','StereoPattern','LEFT_{}.bmp')
scene_RIGHT_PATTERN_path = os.path.join(data_path, obj, 'Scene_{}','{}','StereoPattern','RIGHT_{}.bmp')
########################################################################################################


for scene_id in scene_ids:
    print(scene_id)

    # Load all views
    scene_list_file = scene_list_path.format(scene_id, sensor)
    view_list = inout.load_img_list(scene_list_file)
    for view in view_list:
        #print(view)
        # load Depth Img and Camera Pose
        DEPTH_path = scene_DEPTH_path.format(scene_id, sensor, view)
        depth_img = cv2.imread(DEPTH_path, cv2.IMREAD_ANYDEPTH) * depth_unit
        # convert depth image to disparity map
        disparity_map = inout.convertDepthToDisparity(depth_img, depth_fx, left_cx, right_cx,
                                                      baseline, disparity_shift)

        # load Pattern Stereo Img
        LEFT_PATTERN_path = scene_LEFT_PATTERN_path.format(scene_id, sensor, view)
        RIGHT_PATTERN_path = scene_RIGHT_PATTERN_path.format(scene_id, sensor, view)
        left_pattern_img = cv2.imread(LEFT_PATTERN_path, cv2.IMREAD_GRAYSCALE)
        right_pattern_img = cv2.imread(RIGHT_PATTERN_path, cv2.IMREAD_GRAYSCALE)

        # Find Left-Right Correspondence for Pattern Projected Stereo
        pix_left_x = 211 # Example
        pix_y = 372 # Example
        disparity = disparity_map[pix_y, pix_left_x] # Get disparity value
        if (disparity != 0):
            intensity_left = left_pattern_img[pix_y, pix_left_x]
            pix_right_x = pix_left_x + disparity
            intensity_right = right_pattern_img[pix_y, pix_right_x]



