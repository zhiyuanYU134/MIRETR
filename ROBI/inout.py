import ruamel.yaml as yaml
import numpy as np
import json
import cv2

def load_img_list(scene_list_file):
    try:
        all_file_names = []
        with open(scene_list_file) as f:
            mylist = f.read().splitlines()
        for item in mylist:
            all_file_names.append(item)

        return all_file_names
    except:
        print(scene_list_file, "  does not exist!!!")
        return []

def load_cam_info(path):
    try:
        with open(path, 'r') as f:
            cam_info = yaml.load(f, Loader=yaml.CLoader)
            for eid in cam_info.keys():
                if eid =='distortion':
                    cam_info[eid] = np.array(cam_info[eid])
                    a = 1

        return cam_info
    except:
        return None

def load_camPose_info(scene_file):
    # Camera Pose: the pose of camera in world frame
    try:
        with open(scene_file, "r") as f:
            data = json.load(f)
            cam_pose = np.array(data['cam_pose']).reshape(4,4)
        return cam_pose

    except:
        return []

def load_objPose(gt_file):
    all_gt_poses = []
    try:
        with open(gt_file, "r") as f:
            all_gt_poses = json.load(f)
            for key in all_gt_poses.keys():
                all_gt_poses[key] = np.array(all_gt_poses[key]).reshape(4,4)

        return all_gt_poses
    except:
        print(gt_file, "  does not exist!!!")
        return all_gt_poses

def convertDepthToDisparity(depth_map, f, cx_left, cx_right, baseline, disparity_shift):
    cv_calib_disparity = (cx_right - cx_left) / baseline
    disparity_map = (f / depth_map - cv_calib_disparity) * baseline + disparity_shift
    disparity_map[depth_map==0] = 0

    return -disparity_map