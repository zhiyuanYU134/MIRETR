# ROBI_Toolkit
Python scripts to work on the ROBI dataset - a multi-view dataset for reflective objects in robotic bin-picking.\
ROBI dataset is available at: https://www.trailab.utias.utoronto.ca/robi.
![ROBI](https://github.com/junyang224/ROBI_Toolkit/blob/main/robi.png)


# Requirements
Python >=3.5 \
opencv-python >= 3.1 \
numpy\
ruamel.yaml

# Baseline Methods
We provide the evaluation results on three object pose estimators (PPF [1], Line2D [2], AAE [3]), reasearchers are welcome to compare them against their our methods. The raw evaluation results can be downloaded [here](https://drive.google.com/file/d/1Ru3fmcYFBGOufGUp2jCkFaQgLCv7spIh/view?usp=sharing). Please run "**eval_baselines.py**" to load the the estimated object poses from these baseline approaches.

# Code
 * "**read_scene_imgs.py**": A script to load test images (with 6D camera poses and the ground truth 6D object poses).
 * "**read_pattern_imgs.py**": A script to load stereo pattern images and disparity maps. 
 * "**eval_baselines.py**": A script to load ground truth and the estimated object poses from provided baseline methods.

# Ground Truth Depth Map
Please note that, we capture the ground truth depth map with only Ensenso camera (no Realsense data). 
 * For Scene 4, 5, 8, 9, each viewpoint image has the corresponding GT depth map (in "GT_Depth" folder). 
 * For Scene 1, 2, 3, 6, 7, the GT depth maps were captured only for a subset of viewpoints in Ensenso data folder: 
   - **Scene 1-3**: DEPTH_view_{71-87}
   - **Scene 6-7**: DEPTH_view_{12-14, 16, 18, 22, 24-41}

# Author
Jun Yang\
junyang.yang@mail.utoronto.ca\
Institute for Aerospace Studies, University of Toronto

# References
[1] Drost, Bertram, et al. "Model globally, match locally: Efficient and robust 3D object recognition." 2010 IEEE computer society conference on computer vision and pattern recognition. Ieee, 2010.\
[2] Hinterstoisser, Stefan, et al. "Model based training, detection and pose estimation of texture-less 3d objects in heavily cluttered scenes." Asian conference on computer vision. Springer, Berlin, Heidelberg, 2012.\
[3] Sundermeyer, Martin, et al. "Implicit 3d orientation learning for 6d object detection from rgb images." Proceedings of the european conference on computer vision (ECCV). 2018.

# Citation
If you find ROBI dataset useful in your work, please consider citing:

    @inproceedings{yang2021robi,
      title={ROBI: A Multi-View Dataset for Reflective Objects in Robotic Bin-Picking},
      author={Yang, Jun and Gao, Yizhou and Li, Dong and Waslander, Steven L},
      booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
      pages={9788--9795},
      year={2021},
      organization={IEEE}
    }
    
    @article{yang2021probabilistic,
      title={Probabilistic Multi-View Fusion of Active Stereo Depth Maps for Robotic Bin-Picking},
      author={Yang, Jun and Li, Dong and Waslander, Steven L},
      journal={IEEE Robotics and Automation Letters},
      volume={6},
      number={3},
      pages={4472--4479},
      year={2021},
      publisher={IEEE}
    }
