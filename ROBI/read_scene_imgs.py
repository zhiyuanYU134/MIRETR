import cv2
import numpy as np
import os
import inout

np.set_printoptions(precision=8, suppress=True)

# Dataset Info
######################################
# Path to the ROBI dataset
data_path = '/home/yzy/Desktop/robi/'
#data_path = 'D:/ROBI/'
obj = 'Gear' # Object Name
# Camera sensor to use
sensor = 'Ensenso' #'RealSense'
# Scene ID
scene_ids = [1,2,3,4,5,6,7,8,9]
######################################



# Camera Info
#######################################################
cam_info_path = os.path.join('./ROBI/cam', '{}.yml')
cam_DEPTH_path = cam_info_path.format(sensor+'_DEPTH')
cam_LEFT_path = cam_info_path.format(sensor+'_LEFT')
cam_RIGHT_path = cam_info_path.format(sensor+'_RIGHT')
cam_RGB_path = cam_info_path.format(sensor+'_RGB')

depth_info = inout.load_cam_info(cam_DEPTH_path)
left_info = inout.load_cam_info(cam_LEFT_path)
right_info = inout.load_cam_info(cam_RIGHT_path)
RGB_info = inout.load_cam_info(cam_RGB_path)

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
RGB_fx = None
RGB_fy = None
RGB_cx = None
RGB_cy = None
RGB_distortion = None
if sensor == 'RealSense':
    RGB_fx = RGB_info['fx']
    RGB_fy = RGB_info['fy']
    RGB_cx = RGB_info['cx']
    RGB_cy = RGB_info['cy']
    RGB_distortion = RGB_info['distortion']
#######################################################



# Data path Info
########################################################################################################
# For all cameras
scene_GT_path = os.path.join(data_path, obj, 'Scene_{}','GT_world2obj.json')
scene_list_path = os.path.join(data_path, obj, 'Scene_{}','{}','list.txt')
scene_DEPTH_path = os.path.join(data_path, obj, 'Scene_{}','{}','Depth','DEPTH_{}.png')
scene_DEPTH_camPose = os.path.join(data_path, obj, 'Scene_{}','{}','Depth','DEPTH_{}.json')
# For Ensenso only
scene_LEFT_path = os.path.join(data_path, obj, 'Scene_{}','{}','Stereo','LEFT_{}.bmp')
scene_LEFT_camPose = os.path.join(data_path, obj, 'Scene_{}','{}','Stereo','LEFT_{}.json')
scene_RIGHT_path = os.path.join(data_path, obj, 'Scene_{}','{}','Stereo','RIGHT_{}.bmp')
scene_RIGHT_camPose = os.path.join(data_path, obj, 'Scene_{}','{}','Stereo','RIGHT_{}.json')
# For RealSense only
scene_RGB_path = os.path.join(data_path, obj, 'Scene_{}','{}','Color','RGB_{}.bmp')
scene_RGB_camPose = os.path.join(data_path, obj, 'Scene_{}','{}','Color','RGB_{}.json')
########################################################################################################


for scene_id in scene_ids:
    print(scene_id)
    # Load GT poses for the scene (pose of objects in world coordinate)
    scene_GT_path = scene_GT_path.format(scene_id)
    all_gt_poses = inout.load_objPose(scene_GT_path)

    # Load all views
    scene_list_file = scene_list_path.format(scene_id, sensor)
    view_list = inout.load_img_list(scene_list_file)
    for view in view_list:
        #print(view)
        # load Depth Img and Camera Pose
        DEPTH_path = scene_DEPTH_path.format(scene_id, sensor, view)
        depth_camPose_path = scene_DEPTH_camPose.format(scene_id, sensor, view)
        depth_camPose = inout.load_camPose_info(depth_camPose_path)
        depth_img = cv2.imread(DEPTH_path, cv2.IMREAD_ANYDEPTH) * depth_unit
        # For different camera
        if sensor == 'Ensenso':
            # Load Stereo Img for Ensenso
            LEFT_path = scene_LEFT_path.format(scene_id, sensor, view)
            RIGHT_path = scene_RIGHT_path.format(scene_id, sensor, view)
            LEFT_camPose_path = scene_LEFT_camPose.format(scene_id, sensor, view)
            RIGHT_camPose_path = scene_RIGHT_camPose.format(scene_id, sensor, view)

            left_img = cv2.imread(LEFT_path, cv2.IMREAD_GRAYSCALE)
            right_img = cv2.imread(RIGHT_path, cv2.IMREAD_GRAYSCALE)
            left_camPose = inout.load_camPose_info(LEFT_camPose_path)
            right_camPose = inout.load_camPose_info(RIGHT_camPose_path)
        if sensor == 'RealSense':
            # Load RGB Img for RealSense
            RGB_path = scene_RGB_path.format(scene_id, sensor, view)
            RGB_camPose_path = scene_RGB_camPose.format(scene_id, sensor, view)

            RGB_img = cv2.imread(RGB_path, cv2.IMREAD_COLOR)
            RGB_camPose = inout.load_camPose_info(RGB_camPose_path)
