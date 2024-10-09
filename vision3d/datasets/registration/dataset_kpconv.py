import os.path as osp
import pickle
import random
from functools import partial

import torch
import torch.utils.data
import numpy as np
import pandas as pd
import json,pickle
from pathlib import Path
from PIL import Image
import os
from ROBI import inout

from ...utils.point_cloud_utils import (
    random_sample_rotation, apply_transform
)
from ...utils.registration_utils import get_corr_indices
from ...modules.kpconv.helpers import generate_input_data, calibrate_neighbors
import open3d as o3d
from ...utils.registration_utils import to_o3d_pcd,compute_relative_rotation_error

from vision3d.datasets.registration.transform import Transform
import quaternion





def pairwise_distance(points0, points1, normalized=False, clamp=False):
    r"""
    [PyTorch/Numpy] Pairwise distance of two point clouds.

    :param points0: torch.Tensor (d0, ..., dn, num_point0, num_feature)
    :param points1: torch.Tensor (d0, ..., dn, num_point1, num_feature)
    :param normalized: bool (default: False)
        If True, the points are normalized, so a2 and b2 both 1. This enables us to use 2 instead of a2 + b2 for
        simplicity.
    :param clamp: bool (default: False)
        If True, all value will be assured to be non-negative.
    :return: dist: torch.Tensor (d0, ..., dn, num_point0, num_point1)
    """
    if isinstance(points0, torch.Tensor):
        if len(points0.shape)==3:
            
            dist2 = (points0.unsqueeze(2)-points1.unsqueeze(1)).abs().pow(2).sum(-1)
        else:
            dist2 = (points0.unsqueeze(1)-points1.unsqueeze(0)).abs().pow(2).sum(-1)
    else:
        ab = np.matmul(points0, points1.transpose(-1, -2))
        if normalized:
            dist2 = 2 - 2 * ab
        else:
            a2 = np.expand_dims(np.sum(points0 ** 2, axis=-1), axis=-1)
            b2 = np.expand_dims(np.sum(points1 ** 2, axis=-1), axis=-2)
            dist2 = a2 - 2 * ab + b2
        if clamp:
            dist2 = np.maximum(dist2, np.zeros_like(dist2))
    return dist2


def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M 


def calc_Mbbox(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M



def vanish(Mbbox, scan_warped):
    Mbbox_inverse = np.linalg.inv(Mbbox)
    scan_warped_warped = np.dot(Mbbox_inverse, scan_warped.T).T[:, :3]
    idx = ((np.multiply((scan_warped_warped < 1.1), (scan_warped_warped > -1.1))).sum(-1) < 3).nonzero()[0]

    return idx

def remap_bop_targets(targets):
    targets = targets.rename(columns={'im_id': 'view_id'})
    targets['label'] = targets['obj_id'].apply(lambda x: f'obj_{x:06d}')
    return targets


def build_index(ds_dir, save_file, split, save_file_annotations):
    scene_ids, cam_ids, view_ids = [], [], []

    annotations = dict()
    base_dir = ds_dir / split

    for scene_dir in base_dir.iterdir():
        scene_id = scene_dir.name
        annotations_scene = dict()
        for f in ('scene_camera.json', 'scene_gt_info.json', 'scene_gt.json'):
            path = (scene_dir / f)
            if path.exists():
                annotations_scene[f.split('.')[0]] = json.loads(path.read_text())
        annotations[scene_id] = annotations_scene
        # for view_id in annotations_scene['scene_gt_info'].keys():
        for view_id in annotations_scene['scene_camera'].keys():
            cam_id = 'cam'
            scene_ids.append(int(scene_id))
            cam_ids.append(cam_id)
            view_ids.append(int(view_id))

    frame_index = pd.DataFrame({'scene_id': scene_ids, 'cam_id': cam_ids,
                                'view_id': view_ids, 'cam_name': cam_ids})
    frame_index.to_feather(save_file)
    save_file_annotations.write_bytes(pickle.dumps(annotations))
    return

def rotation_matrix(num_axis, augment_rotation):
    """
    Sample rotation matrix along [num_axis] axis and [0 - augment_rotation] angle
    Input
        - num_axis:          rotate along how many axis
        - augment_rotation:  rotate by how many angle
    Output
        - R: [3, 3] rotation matrix
    """
    assert num_axis == 1 or num_axis == 3 or num_axis == 0
    if num_axis == 0:
        return np.eye(3)
    angles = np.random.rand(3) * 2 * np.pi * augment_rotation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    # R = Rx @ Ry @ Rz
    if num_axis == 1:
        return random.choice([Rx, Ry, Rz])
    return Rx @ Ry @ Rz


def translation_matrix(augment_translation):
    """
    Sample translation matrix along 3 axis and [augment_translation] meter
    Input
        - augment_translation:  translate by how many meters
    Output
        - t: [3, 1] translation matrix
    """
    T = np.random.rand(3) * augment_translation
    return T.reshape(3, 1)


def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0, 2, 1) + trans[:, :3, 3:4]
        return trans_pts.permute(0, 2, 1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T

class ThreeDMatchPairKPConvDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_root,
                 split,
                 matching_radius,
                 max_point=30000,
                 use_augmentation=True,
                 augmentation_noise=0.005,
                 rotation_factor=1,
                 overlap_thresh=None,
                 return_correspondences=True,
                 suffix=None,
                 aligned=False,
                 rotated=False):
        super(ThreeDMatchPairKPConvDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = split
        self.matching_radius = matching_radius
        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.rotation_factor = rotation_factor
        self.max_point = max_point
        self.return_correspondences = return_correspondences
        self.suffix = suffix
        self.aligned = aligned
        self.rotated = rotated

        ds_dir = Path(dataset_root)
        self.ds_dir = ds_dir
        assert ds_dir.exists(), 'Dataset does not exists.'

        self.split = split
        self.base_dir = ds_dir / self.split

        save_file_index = self.ds_dir / f'index_{self.split}.feather'
        save_file_annotations = self.ds_dir / f'annotations_{self.split}.pkl'
        build_index(ds_dir=ds_dir, save_file=save_file_index,save_file_annotations=save_file_annotations,split=self.split)
        
        self.frame_index = pd.read_feather(save_file_index).reset_index(drop=True)
        self.annotations = pickle.loads(save_file_annotations.read_bytes())

        models_infos = json.loads((ds_dir / 'models' / 'models_info.json').read_text())
        self.all_labels = [f'obj_{int(obj_id):06d}' for obj_id in models_infos.keys()]

        objects = []
        for obj_id, bop_info in models_infos.items():
            obj_id = int(obj_id)
            obj_label = f'obj_{obj_id:06d}'
            mesh_path = (ds_dir / 'models' / obj_label).with_suffix('.ply').as_posix()
            obj = dict(
                label=obj_label,
                category=None,
                mesh_path=mesh_path,
                mesh_units='mm',
            )
            is_symmetric = False
            for k in ('symmetries_discrete', 'symmetries_continuous'):
                obj[k] = bop_info.get(k, [])
                if len(obj[k]) > 0:
                    is_symmetric = True
            obj['is_symmetric'] = is_symmetric
            obj['diameter'] = bop_info['diameter']
            scale = 0.001 if obj['mesh_units'] == 'mm' else 1.0
            obj['diameter_m'] = bop_info['diameter'] * scale
            objects.append(obj)
        self.objects = objects

    def __len__(self):
        return len(self.frame_index)

    def _augment_point_cloud(self, points0, points1, Rts):
        aug_rotation = random_sample_rotation(self.rotation_factor)
        if random.random() > 0.5:
            points0 = np.matmul(points0, aug_rotation.T)
            for i in range(len(Rts)):
                Rt=Rts[i]
                rotation=Rt[:3, :3]
                translation=Rt[:3, 3]
                rotation = np.matmul(aug_rotation, rotation)
                translation = np.matmul(aug_rotation, translation)
                Rt[:3, :3] = rotation
                Rt[:3, 3] = translation
                Rts[i]=Rt
        else:
            points1 = np.matmul(points1, aug_rotation.T)
            for i in range(len(Rts)):
                Rt=Rts[i]
                rotation=Rt[:3, :3]
                translation=Rt[:3, 3]
                rotation = np.matmul(rotation, aug_rotation.T)
                Rt[:3, :3] = rotation
                Rt[:3, 3] = translation
                Rts[i]=Rt
        """ points0 += (np.random.rand(points0.shape[0], 3) - 0.5) * self.augmentation_noise
        points1 += (np.random.rand(points1.shape[0], 3) - 0.5) * self.augmentation_noise """
        return points0, points1, Rts

    def __getitem__(self, frame_id):
        # metadata

        row = self.frame_index.iloc[frame_id]
        scene_id, view_id = row.scene_id, row.view_id
        view_id = int(view_id)
        view_id_str = f'{view_id:06d}'
        scene_id_str = f'{int(scene_id):06d}'
        scene_dir = self.base_dir / scene_id_str

        depth_dir =scene_dir/ 'depth'
        depth_path = depth_dir / f'{view_id_str}.png'
        rgb = np.array(Image.open(depth_path))

        depth_raw = o3d.io.read_image(str(depth_path))
        inter = o3d.camera.PinholeCameraIntrinsic()
        cam_annotation = self.annotations[scene_id_str]['scene_camera'][str(view_id)]

        if 'cam_R_w2c' in cam_annotation:
            RC0 = np.array(cam_annotation['cam_R_w2c']).reshape(3, 3)
            tC0 = np.array(cam_annotation['cam_t_w2c']) * 0.001
            TC0 = Transform(RC0, tC0)
        else:
            TC0 = Transform(np.eye(3), np.zeros(3))
        K = np.array(cam_annotation['cam_K']).reshape(3, 3)
        T0C = TC0.inverse()
        T0C = T0C.toHomogeneousMatrix()
        inter.set_intrinsics(rgb.shape[0], rgb.shape[1], K[0][0], K[1][1], K[0][2], K[1][2])
        T0C = TC0.inverse()
        #mask = np.zeros((h, w), dtype=np.uint8)
        if 'scene_gt_info' in self.annotations[scene_id_str]:
            annotation = self.annotations[scene_id_str]['scene_gt'][str(view_id)]
            n_objects = len(annotation)
            visib = self.annotations[scene_id_str]['scene_gt_info'][str(view_id)]

            pcd0 = o3d.geometry.PointCloud.create_from_depth_image( depth_raw, inter,depth_scale=1000) 
            pcd0=pcd0.voxel_down_sample(voxel_size=0.005)       
            points0 =  np.array(pcd0.points).astype(np.float32)
            feats0 = np.ones((points0.shape[0], 1), dtype=np.float32)
            obj_id = annotation[0]['obj_id']-1
            pcd1=o3d.io.read_point_cloud(self.objects[obj_id]['mesh_path'])
            pcd1=pcd1.voxel_down_sample(voxel_size=2.5)       
            points1 =  np.array(pcd1.points).astype(np.float32)/1000
            feats1 = np.ones((points1.shape[0], 1), dtype=np.float32)
            Rts=[]
            correspondences=[]
            for n in range(n_objects):
                RCO = np.array(annotation[n]['cam_R_m2c']).reshape(3, 3)
                tCO = np.array(annotation[n]['cam_t_m2c']) * 0.001
                TCO = Transform(RCO, tCO)
                T0O = T0C * TCO
                T0O = T0O.toHomogeneousMatrix()
                Rts.append(T0O.astype(np.float32))
                correspondences.append(get_corr_indices(points0, points1, T0O.astype(np.float32), self.matching_radius))
            scene_name=scene_id_str
            frag_id0=view_id
            frag_id1=obj_id
        if self.use_augmentation:
            points0, points1, Rts = self._augment_point_cloud(points0, points1, Rts)   
        return points0,points1,feats0,feats1,Rts,correspondences,scene_name, frag_id0, frag_id1


class ROBISENCEDataset(torch.utils.data.Dataset):
    def __init__(self,
                vox,
                 dataset_root,
                 split,
                 matching_radius,
                 max_point=30000,
                 use_augmentation=True,
                 augmentation_noise=0.005,
                 rotation_factor=1,
                 overlap_thresh=None,
                 return_correspondences=True,
                 suffix=None,
                 aligned=False,
                 rotated=False):
        super(ROBISENCEDataset, self).__init__()
        self.matching_radius = matching_radius
        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.rotation_factor = rotation_factor
        self.max_point = max_point
        self.return_correspondences = return_correspondences
        self.suffix = suffix
        self.aligned = aligned
        self.rotated = rotated
        self.voxel_size=vox
        
        self.obj_names = ['DSub_connector','Chrome_screw' ,'Tube_fitting','DIN_connector','Gear', 'Eye_bolt' ,'Zigzag']#
        self.sym ={}#
        self.sym['DSub_connector']="__SYM_NONE"
        self.sym['DIN_connector']="__SYM_NONE"
        self.sym['Zigzag']="__SYM_NONE" 
        self.sym['Chrome_screw']="__SYM"
        self.sym['Tube_fitting']="__SYM"
        self.sym['Gear']="__SYM"
        self.sym['Eye_bolt']="__SYM"
        self.data_path=dataset_root
        self.obj_path=dataset_root+'Object_models/'
        self.sensor = 'Ensenso' #'RealSense'
        self.train_scene_ids = [1,2,3,6,7,8]
        self.test_scene_ids = [5,9]
        self.val_scene_ids = [4]
        
        cam_info_path = os.path.join('./ROBI/cam', '{}.yml')
        cam_DEPTH_path = cam_info_path.format(self.sensor+'_DEPTH')
        
        depth_info = inout.load_cam_info(cam_DEPTH_path)

        self.depth_unit = 0.03125#depth_info['depth_unit']
        self.depth_fx = 1083.097046#depth_info['fx']
        self.depth_fy = 1083.097046#depth_info['fy']
        self.depth_cx = 379.326874#depth_info['cx']
        self.depth_cy = 509.437195#depth_info['cy']
        self.frame_indexs=[]

        self.partition = split
        if self.partition == 'train':
            for obj in self.obj_names:
                for scene_id in self.train_scene_ids:
                    scene_list_path = os.path.join(self.data_path, obj, 'Scene_{}','{}','list.txt')
                    scene_list_file = scene_list_path.format(scene_id, self.sensor)
                    view_list = inout.load_img_list(scene_list_file)
                    for view in view_list:
                        self.frame_indexs.append([obj,scene_id,view])
            random.shuffle(self.frame_indexs)
            self.num = len(self.frame_indexs)
            self.start = 0
            self.use_augmentation = True
            self.augmentation_noise = augmentation_noise
            self.rotation_factor = rotation_factor

        elif self.partition == 'val':
            for obj in self.obj_names:
                for scene_id in self.val_scene_ids:
                    scene_list_path = os.path.join(self.data_path, obj, 'Scene_{}','{}','list.txt')
                    scene_list_file = scene_list_path.format(scene_id, self.sensor)
                    view_list = inout.load_img_list(scene_list_file)
                    for view in view_list:
                        self.frame_indexs.append([obj,scene_id,view])
            self.num = len(self.frame_indexs)
            self.start = 0
            self.use_augmentation = False

        elif self.partition == 'test':
            for obj in self.obj_names:
                for scene_id in self.test_scene_ids:
                    scene_list_path = os.path.join(self.data_path, obj, 'Scene_{}','{}','list.txt')
                    scene_list_file = scene_list_path.format(scene_id, self.sensor)
                    view_list = inout.load_img_list(scene_list_file)
                    for view in view_list:
                        self.frame_indexs.append([obj,scene_id,view])
            #random.shuffle(self.frame_indexs)
            self.num = len(self.frame_indexs)
            self.start = 0
            self.use_augmentation = False
        else:
            print('gg')

    def __len__(self):
        return self.num

    def _augment_point_cloud(self, points0, points1, Rts):
        aug_rotation = random_sample_rotation(self.rotation_factor)
        if random.random() > 0.5:
            points0 = np.matmul(points0, aug_rotation.T)
            for i in range(len(Rts)):
                Rt=Rts[i]
                rotation=Rt[:3, :3]
                translation=Rt[:3, 3]
                rotation = np.matmul(aug_rotation, rotation)
                translation = np.matmul(aug_rotation, translation)
                Rt[:3, :3] = rotation
                Rt[:3, 3] = translation
                Rts[i]=Rt
        else:
            points1 = np.matmul(points1, aug_rotation.T)
            for i in range(len(Rts)):
                Rt=Rts[i]
                rotation=Rt[:3, :3]
                translation=Rt[:3, 3]
                rotation = np.matmul(rotation, aug_rotation.T)
                Rt[:3, :3] = rotation
                Rt[:3, 3] = translation
                Rts[i]=Rt
        return points0, points1, Rts
    
    def  normal_redirect(self,points, normals, view_point):
        '''
        Make direction of normals towards the view point
        '''
        vec_dot = np.sum((view_point - points) * normals, axis=-1)
        mask = (vec_dot < 0.)
        redirected_normals = normals.copy()
        redirected_normals[mask] *= -1.
        return redirected_normals

    def __getitem__(self, frame_id):
        # metadata
        id =self.start+frame_id
        frame_index=self.frame_indexs[id]
        obj=frame_index[0]
        scene_id=frame_index[1]
        view=frame_index[2]
        scene_GT_path = os.path.join(self.data_path, obj, 'Scene_{}','GT_world2obj.json')
        scene_DEPTH_path = os.path.join(self.data_path, obj, 'Scene_{}','{}','Depth','DEPTH_{}.png')
        scene_DEPTH_camPose = os.path.join(self.data_path, obj, 'Scene_{}','{}','Depth','DEPTH_{}.json')

        scene_GT_path = scene_GT_path.format(scene_id)
        all_gt_poses = inout.load_objPose(scene_GT_path)
           
        DEPTH_path = scene_DEPTH_path.format(scene_id, self.sensor, view)
        depth_camPose_path = scene_DEPTH_camPose.format(scene_id, self.sensor, view)
        jsonList = []
        with open(depth_camPose_path, "r") as f:
            for jsonObj in f:
                data = json.loads(jsonObj)
                jsonList.append(data)
        depth_camPose = np.array(jsonList[0]['cam_pose']).reshape(4,4)
        RC0 = depth_camPose[:3,:3]
        tC0 =depth_camPose[:3,3] /1000#* depth_unit
        TC0 = Transform(RC0, tC0)
        T0C = TC0.inverse()
        T0C = T0C.toHomogeneousMatrix()
        T0C = TC0.inverse()
        depth_raw = o3d.io.read_image(str(DEPTH_path))
        inter = o3d.camera.PinholeCameraIntrinsic()
        rgb = np.array(Image.open(DEPTH_path))
        inter.set_intrinsics(rgb.shape[0], rgb.shape[1], self.depth_fx, self.depth_fy, self.depth_cx,self.depth_cy)
        pcd0 = o3d.geometry.PointCloud.create_from_depth_image( depth_raw, inter,depth_scale=1.0/self.depth_unit) 
        points=np.array(pcd0.points).astype(np.float32)/1000
        point0_ori=points
        pcd0=to_o3d_pcd(points)
        pcd0=pcd0.voxel_down_sample(voxel_size=self.voxel_size)  
        points0 =  np.array(pcd0.points).astype(np.float32)

        pcd1 = o3d.io.read_triangle_mesh(self.obj_path+obj+'.stl')
        #pcd1.compute_vertex_normals()
        points=np.array(pcd1.vertices).astype(np.float32)/1000
        point1_ori=points
        pcd1=to_o3d_pcd(points)
        pcd1=pcd1.voxel_down_sample(voxel_size=self.voxel_size)  
        points1 =  np.array(pcd1.points).astype(np.float32)
        Rts=[]
        correspondences=[]
        rres=[]
        lengths=float(points1.shape[0])
        for gt_pose in all_gt_poses.values():
            RCO = gt_pose[:3,:3]
            tCO = gt_pose[:3,3]/1000 #* depth_unit
            TCO = Transform(RCO, tCO)
            T0O = T0C * TCO
            T0O = T0O.toHomogeneousMatrix()
            Rts.append(T0O.astype(np.float32))
            rre=compute_relative_rotation_error(torch.from_numpy(T0O.astype(np.float32)[:3,:3]),torch.eye(3))
            rres.append(rre)
            correspondence=get_corr_indices(points0, points1, T0O.astype(np.float32), self.matching_radius)
            if len(correspondence)==0:
                correspondences.append(len(correspondence)/lengths)
            else:
                correspondence=torch.from_numpy(correspondence[:,1])
                correspondence=torch.unique(correspondence)
                correspondences.append(len(correspondence)/lengths)
        if self.use_augmentation:
            points0, points1, Rts = self._augment_point_cloud(points0, points1, Rts)   
        feats0 = np.ones((points0.shape[0], 1), dtype=np.float32)
        feats1 = np.ones((points1.shape[0], 1), dtype=np.float32)
        return points0,points1,feats0,feats1,Rts,torch.from_numpy(np.array(correspondences)),scene_id, self.sym[obj], frame_index,point0_ori,points1


class Scan2cadKPConvDataset(torch.utils.data.Dataset):
    def __init__(self,
                vox_size,
                 scannet_root,
                 shapenet_root,
                 scan2cad_root,
                 split,
                 matching_radius,
                 max_point=30000,
                 use_augmentation=True,
                 augmentation_noise=0.005,
                 rotation_factor=1,
                 overlap_thresh=None,
                 return_correspondences=True,
                 suffix=None,
                 aligned=False,
                 rotated=False):
        super(Scan2cadKPConvDataset, self).__init__()

        self.scannet_root = scannet_root
        self.shapenet_root = shapenet_root
        self.scan2cad_root = scan2cad_root
        self.partition = split
        self.matching_radius = matching_radius
        
        self.max_point = max_point
        self.return_correspondences = return_correspondences
        self.suffix = suffix
        self.aligned = aligned
        self.rotated = rotated
        self.scan2cad=np.load(scan2cad_root,allow_pickle=True)
        self.train_num = 1528
        self.val_num = 218
        self.test_num = 438
        self.vox_size=vox_size
        if self.partition == 'train':
            self.num = self.train_num
            self.start = 0
            self.use_augmentation = True
            self.augmentation_noise = augmentation_noise
            self.rotation_factor = rotation_factor

        elif self.partition == 'val':
            self.num = self.val_num
            self.start = self.train_num
            self.use_augmentation = False

        elif self.partition == 'test':
            self.num = self.test_num
            self.start = self.train_num + self.val_num
            self.use_augmentation = False
        elif self.partition == 'all':
            self.start = 0
            self.num = self.test_num+self.train_num + self.val_num
            self.use_augmentation = False
        else:
            print('gg')

    def __len__(self):
        return self.num
    
    def _augment_point_cloud(self, points0, points1, Rts):
        aug_rotation = random_sample_rotation(self.rotation_factor)
        if random.random() > 0.5:
            points0 = np.matmul(points0, aug_rotation.T)
            for i in range(len(Rts)):
                Rt=Rts[i]
                rotation=Rt[:3, :3]
                translation=Rt[:3, 3]
                rotation = np.matmul(aug_rotation, rotation)
                translation = np.matmul(aug_rotation, translation)
                Rt[:3, :3] = rotation
                Rt[:3, 3] = translation
                Rts[i]=Rt
        else:
            points1 = np.matmul(points1, aug_rotation.T)
            for i in range(len(Rts)):
                Rt=Rts[i]
                rotation=Rt[:3, :3]
                translation=Rt[:3, 3]
                rotation = np.matmul(rotation, aug_rotation.T)
                Rt[:3, :3] = rotation
                Rt[:3, 3] = translation
                Rts[i]=Rt
        return points0, points1, Rts


    def __getitem__(self, id):
        
        # metadata
        frame_id=self.start+id
        dataset=self.scan2cad[frame_id]
        id_scan = dataset['id_scan']
        trans_scan = dataset['trans_scan']
        scan_root = self.scannet_root  + '/'+ id_scan + '/' + id_scan + '_vh_clean_2.ply'
        
        pcd = o3d.io.read_point_cloud(scan_root)
        Mscan = make_M_from_tqs(trans_scan['translation'], trans_scan['rotation'], trans_scan['scale'])
        scan_homo = np.concatenate((np.array(pcd.points), np.ones([np.array(pcd.points).shape[0], 1])), axis=-1)
        scan_warped = np.dot(Mscan, scan_homo.T).T

        scale_min = []
        for cad in dataset['cad']:
            scale_min.append(cad['trs']['scale'])
            id_cad = cad["id_cad"]
            catid_cad = cad["catid_cad"]
            sym=cad["sym"]
        cadroot = self.shapenet_root  + '/'+ catid_cad + '/' + id_cad +'/models'+ '/model_normalized.obj'
        scale_min = np.array(scale_min).min(0).tolist()
        cad = o3d.io.read_triangle_mesh(cadroot)
        cad = cad.sample_points_uniformly(10000)
        
        cad_homo = np.concatenate((np.array(cad.points), np.ones([np.array(cad.points).shape[0], 1])), axis=-1)

        T = np.eye(4)
        R = np.eye(4)
        S = np.eye(4)
        S[0:3, 0:3] = np.diag(scale_min)

        points1 = np.matmul(T.dot(R).dot(S), cad_homo.T).T[:, :3]
        cad_homo = np.concatenate((np.array(cad.points), np.ones([np.array(cad.points).shape[0], 1])), axis=-1)
        point1_ori=points1
        
        cad=to_o3d_pcd(points1)
        cad=cad.voxel_down_sample(voxel_size=self.vox_size)  
        points1=np.array(cad.points)
        trans = []

        for model in dataset['cad']:
            t = model["trs"]["translation"]
            q = model["trs"]["rotation"]
            s = [1, 1, 1]

            Mcad = make_M_from_tqs(t, q, s)
            Mcad_min = make_M_from_tqs(t, q, scale_min)
            Mbbox = calc_Mbbox(model)
            trans.append(Mcad)
            
            idx = vanish(Mbbox, scan_warped)

            cad_warped = np.matmul(Mcad_min, cad_homo.T).T
            scan_part = scan_warped[idx]
            scan_warped = np.concatenate((scan_part, cad_warped), axis=0)
        Rts = np.array(trans)
        points0 = scan_warped[:, :3]
        np.random.shuffle(points0)
        point0_ori=points0
        pcd=to_o3d_pcd(points0)
        pcd=pcd.voxel_down_sample(voxel_size=self.vox_size)  
        points0 = np.array(pcd.points)
        if self.use_augmentation:
            points0, points1, Rts = self._augment_point_cloud(points0, points1, Rts)   
    
        feats0 = np.ones((points0.shape[0], 1), dtype=np.float32)
        feats1 = np.ones((points1.shape[0], 1), dtype=np.float32)
        correspondences=[]
    
        return points0,points1,feats0,feats1,Rts,correspondences, frame_id, sym,id_cad ,point0_ori,point1_ori



class Process_Scan2cadKPConvDataset(torch.utils.data.Dataset):
    def __init__(self,
                 scan2cad_root,
                 split,
                 max_point=30000,
                 use_augmentation=True,
                 augmentation_noise=0.005,
                 rotation_factor=1,
                 overlap_thresh=None,
                 return_correspondences=True,
                 suffix=None,
                 aligned=False,
                 rotated=False):
        super(Process_Scan2cadKPConvDataset, self).__init__()
        self.vox_size=0.025
        self.scan2cad_root = scan2cad_root
        self.partition = split
        
        self.max_point = max_point
        self.return_correspondences = return_correspondences
        self.suffix = suffix
        self.aligned = aligned
        self.rotated = rotated
        self.train_num = 1528
        self.val_num = 218
        self.test_num = 438
        if self.partition == 'train':
            self.num = self.train_num
            self.start = 0
            self.use_augmentation = True
            self.augmentation_noise = augmentation_noise
            self.rotation_factor = rotation_factor

        elif self.partition == 'val':
            self.num = self.val_num
            self.start = self.train_num
            self.use_augmentation = False

        elif self.partition == 'test':
            self.num = self.test_num
            self.start = self.train_num + self.val_num
            self.use_augmentation = False
        else:
            print('gg')

        
        

    def __len__(self):
        return self.num


    def __getitem__(self, id):
        
        # metadata
        frame_id=self.start+id
        dataset=np.load(self.scan2cad_root+ 'data{:05d}.npz'.format(frame_id))
        points0=dataset['points0']
        points1=dataset['points1']
        pcd=to_o3d_pcd(points0)
        pcd=pcd.voxel_down_sample(voxel_size=self.vox_size)  
        points0 = np.array(pcd.points)

        pcd=to_o3d_pcd(points1)
        pcd=pcd.voxel_down_sample(voxel_size=self.vox_size)  
        points1 = np.array(pcd.points)
        """ if self.max_point is not None and points0.shape[0] > self.max_point:
            indices = np.random.permutation(points0.shape[0])[: self.max_point]
            points0 = points0[indices]
        if self.max_point is not None and points1.shape[0] > self.max_point:
            indices = np.random.permutation(points1.shape[0])[: self.max_point]
            points1 = points1[indices] """
        Rts=dataset['trans']
        frag_id1=dataset['frag_id1']
        scene_name=dataset['scene_name']
        sym=dataset['sym']
        
        feats0 = np.ones((points0.shape[0], 1), dtype=np.float32)
        feats1 = np.ones((points1.shape[0], 1), dtype=np.float32)
        
        data_dict = {}

        correspondences=[]
    
        return points0,points1,feats0,feats1,Rts,correspondences, sym,sym,sym, points0,points1

class SyntheticShapeNet(torch.utils.data.Dataset):
    def __init__(self,
                vox_size,
                 shapenet_root,
                 split,
                 matching_radius,
                 max_point=30000,
                 use_augmentation=True,
                 augmentation_noise=0.005,
                 rotation_factor=1,
                 overlap_thresh=None,
                 return_correspondences=True,
                 suffix=None,
                 aligned=False,
                 rotated=False):
        super(SyntheticShapeNet, self).__init__()

        self.shapenet_root = shapenet_root
        self.partition = split
        self.matching_radius = matching_radius
        
        self.max_point = max_point
        self.return_correspondences = return_correspondences
        self.suffix = suffix
        self.aligned = aligned
        self.rotated = rotated

        self.vox_size=vox_size
        self.index=[]
        self.max_object=500
        self.num_instance = 16
        self.max_instance_drop=12
        self.augment_axis = 3
        self.augment_rotation = 1
        self.augment_translation = 5

        aug_T=torch.zeros((3,9,3))
        aug_T[0,:,2]=-1
        aug_T[1,:,2]=0
        aug_T[2,:,2]=1
        x=torch.linspace(-1,1,3)
        x=x.reshape(-1,1)
        x=x.expand(x.shape[0],3).reshape(x.shape[0],3,1)
        y=torch.linspace(-1,1,3) 
        y=y.reshape(1,-1)
        y=y.expand(3,y.shape[1]).reshape(3,y.shape[1],1)
        xy=torch.cat((x,y),dim=-1)
        xy=xy.reshape(-1,xy.shape[-1])
        aug_T[:,:,:2]=xy
        self.aug_T=aug_T.reshape(-1,3)

        train_index=[1,2,3,4,6,8,9,10,12,14,17,19,22,23,25,28,30,31,32,34,35,39,41,43,45,46,49,50,51,54]
        test_index=[0,5,7,11,13,15,16,18,20,21,24,26,27,29,33,36,37,38,40,42,44,47,48,52,53]#
        train_catgory_dataset_paths=[]
        eval_catgory_dataset_paths=[]
        test_catgory_dataset_paths=[]

        with open(osp.join(self.shapenet_root, 'shapenet_path.pkl'), 'rb') as f:
            self.shapenet_path = pickle.load(f)  
        for i in range(55):
            paths=self.shapenet_path[i]
            if len(paths)>self.max_object:
                paths=paths[:self.max_object]
                eval_paths=paths[self.max_object:]
            if i in train_index:
                train_catgory_dataset_paths+=paths
                if len(paths)>self.max_object:
                    eval_catgory_dataset_paths+=eval_paths
            else:
                test_catgory_dataset_paths+=paths
        eval_catgory_dataset_paths=random.shuffle(eval_catgory_dataset_paths)[:900]
        if self.partition == 'train':
            self.index=train_catgory_dataset_paths
            self.num = len(self.index)
            self.use_augmentation = True
   
        elif self.partition == 'val':

            self.index=eval_catgory_dataset_paths
            self.num = len(self.index)
            self.use_augmentation = True

        elif self.partition == 'test':
            self.index=test_catgory_dataset_paths
            self.num = len(self.index)
            self.use_augmentation = True
        else:
            print('gg')

    def __len__(self):
        return self.num
    def produce_augment(self):
        aug_R = []
        for i in range(self.num_instance):
            aug_R.append(rotation_matrix(self.augment_axis, self.augment_rotation))
        return aug_R
    def integrate_trans(self, R, t):
        """
        Integrate SE3 transformations from R and t, support torch.Tensor and np.ndarry.
        Input
            - R: [3, 3] or [bs, 3, 3], rotation matrix
            - t: [3, 1] or [bs, 3, 1], translation matrix
        Output
            - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
        """
        trans = np.eye(4).reshape(-1,4,4).repeat(self.num_instance,0)
        trans[:, :3, :3] = R
        trans[:, :3, 3:4] = t
        return trans

    def transform(self, pts, trans):
        """
        Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
        Input
            - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
            - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
        Output
            - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
        """
        trans_pts = trans[:, :3, :3] @ pts.transpose(0, 2, 1) + trans[:, :3, 3:4]
        return trans_pts.transpose(0, 2, 1)

    def __getitem__(self, id):
        # metadata
        
        cad_path=os.path.join(self.shapenet_root,self.index[id].split('ShapeNetCore')[-1])

        cad = o3d.io.read_triangle_mesh(cad_path)
        cad = cad.sample_points_uniformly(10000)
        
        cad_points = np.array(cad.points).astype(np.float32)
        src_R=pairwise_distance(cad_points,cad_points).max()
        cad_points/= np.sqrt(src_R) 
        
        cad=to_o3d_pcd(cad_points)
        
        cad=cad.voxel_down_sample(voxel_size=self.vox_size)  
        cad_points = np.array(cad.points).astype(np.float32)
        
        rand_r=torch.from_numpy(rotation_matrix(self.augment_axis, self.augment_rotation))
        trans = torch.eye(4)
        trans[:3, :3] = rand_r
        aug_T = apply_transform(self.aug_T, trans).unsqueeze(-1).numpy()
        inds = np.random.choice(range(27), self.num_instance, replace=False) 
        aug_T=aug_T[inds]

        src_keypts = cad_points.reshape(1,cad_points.shape[0],3).repeat(self.num_instance,0)
        tgt_keypts = src_keypts + np.clip(0.01 * np.random.randn(self.num_instance,cad_points.shape[0],3), -1 * 0.05, 0.05)
        
        aug_R = self.produce_augment()
        aug_trans = self.integrate_trans(aug_R, aug_T)

        tgt_keypts = self.transform(tgt_keypts, aug_trans)
        num_instance_drop = int(self.max_instance_drop * np.random.rand() // 1)
        num_instance_input = int(self.num_instance - num_instance_drop)
        inds = np.random.choice(range(self.num_instance), num_instance_input, replace=False) 
        tgt_keypts=tgt_keypts[inds].reshape(-1,3)
        np.random.shuffle(tgt_keypts)
        np.random.shuffle(cad_points)
        aug_trans=aug_trans[inds].astype(np.float32)
              
        feats0 = np.ones((tgt_keypts.shape[0], 1), dtype=np.float32)
        feats1 = np.ones((cad_points.shape[0], 1), dtype=np.float32)
        correspondences=[]
        
        return tgt_keypts, cad_points, feats0, feats1, trans, correspondences, correspondences, correspondences, correspondences ,tgt_keypts,cad_points

def threedmatch_kpconv_collate_fn(list_data, config, neighborhood_limits, compute_indices=True):
    data_dicts = []

    for points0, points1, feats0, feats1, transform, correspondences, scene_name, frag_id0, frag_id1 ,points0_ori,points1_ori in list_data:
        data_dict = {}
        
        data_dict['scene_name'] = scene_name
        data_dict['frag_id0'] = frag_id0
        data_dict['frag_id1'] = frag_id1
        data_dict['transform'] = torch.tensor(transform, dtype=torch.float32)
        #data_dict['correspondences'] = torch.from_numpy(correspondences)
        data_dict['correspondences'] =correspondences
        data_dict['features'] = torch.from_numpy(np.concatenate([feats0, feats1], axis=0))
        data_dict['ref_points_ori'] = torch.from_numpy(points0_ori)
        data_dict['src_points_ori'] = torch.from_numpy(points1_ori)

        stacked_points = torch.from_numpy(np.concatenate([points0, points1], axis=0))
        stacked_lengths = torch.from_numpy(np.array([points0.shape[0], points1.shape[0]]))

        if compute_indices:
            input_points, input_neighbors, input_pools, input_upsamples, input_lengths = generate_input_data(
                stacked_points, stacked_lengths, config, neighborhood_limits
            )
            ref_length_c = input_lengths[-1][0].item()                
            points_c = input_points[-1].detach()
            ref_points_c = points_c[:ref_length_c]

            data_dict['points'] = input_points
            data_dict['neighbors'] = input_neighbors
            data_dict['pools'] = input_pools
            data_dict['upsamples'] = input_upsamples
            data_dict['stack_lengths'] = input_lengths
        else:
            data_dict['stacked_points'] = stacked_points
            data_dict['stacked_lengths'] = stacked_lengths
        data_dicts.append(data_dict)
        
    if len(data_dicts) == 1:
        return data_dicts[0]
    else:
        return data_dicts


def get_dataloader(
        dataset,
        config,
        batch_size,
        num_workers,
        shuffle=False,
        neighborhood_limits=None,
        drop_last=True,
        sampler=None,
        compute_indices=True
):
    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, config, collate_fn=threedmatch_kpconv_collate_fn)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=partial(
            threedmatch_kpconv_collate_fn, config=config, neighborhood_limits=neighborhood_limits,
            compute_indices=compute_indices
        ),
        drop_last=drop_last
    )
    return dataloader, neighborhood_limits


