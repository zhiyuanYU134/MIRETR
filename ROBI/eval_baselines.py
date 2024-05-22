import cv2
import numpy as np
from numpy.linalg import inv
import os
import inout

np.set_printoptions(precision=8, suppress=True)

# Dataset and Results Info
######################################################
# Path to the ROBI dataset
data_path = '/Add/your/path/here/ROBI/'
#data_path = 'D:/ROBI/'
# Path to the baseline results
result_path = '/Add/your/path/here/Baseline_results/'
#result_path = 'D:/Baseline_results/'
obj = 'Zigzag' # Object Name
# Camera sensor to use
sensor = 'Ensenso' #'RealSense'
# Baseline approach
method = 'AAE' # Line2D, PPF
# Scene ID
scene_id = 5
# Viewpoint
viewpoint = 10
######################################################

# Data path Info
#######################################################################
# The ground truth of object poses are under world frame
GT_path = os.path.join(data_path, obj, 'Scene_{}','GT_world2obj.json')
GT_path = GT_path.format(scene_id)
GT_poses = inout.load_objPose(GT_path)
num_objects_gt = len(GT_poses)
#######################################################################

# Result path Info
#######################################################################################
# The estimatad object poses are under world frame
# world2object = world2camera * camera2object
RESULT_path = os.path.join(result_path, sensor, obj, method, 'Scene{}','view_{}.json')
RESULT_path = RESULT_path.format(scene_id, viewpoint)
ESTIMATED_poses = inout.load_objPose(RESULT_path)
num_objects_est = len(ESTIMATED_poses)
#######################################################################################

# Evaluation: Load object poses
############################################
# Load GT poses
for i in range(0, num_objects_gt):
    gt_id = 'Object_' + str(i + 1)
    gt_pose = GT_poses[gt_id]
    print("GT Pose for", gt_id, ":")
    print(gt_pose, "\n")

# Load estimated poses
for j in range(0, num_objects_est):
    est_id = 'Object_' + str(j + 1)
    est_pose = ESTIMATED_poses[est_id]
    print("Estimated Pose by", method, ":")
    print(est_pose, "\n")
############################################

