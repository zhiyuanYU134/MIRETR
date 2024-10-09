import os
import os.path as osp
import argparse

from easydict import EasyDict as edict
from pytools import F

from vision3d.utils.python_utils import ensure_dir

config = edict()

# random seed
config.seed = 7351
#scan2cad
# dir --test_epoch 29 --benchmark 3DMatch

config.root_dir = '/home/yuzhiyuan/MIRETR-main'
config.working_dir = osp.dirname(osp.realpath(__file__))
config.program_name = osp.basename(config.working_dir)
config.output_dir = osp.join(config.root_dir, 'output', config.program_name)
config.snapshot_dir = osp.join(config.output_dir, 'snapshots')
config.logs_dir = osp.join(config.output_dir, 'logs')
config.events_dir = osp.join(config.output_dir, 'events')
config.features_dir = osp.join(config.output_dir, 'features')
config.registration_dir = osp.join(config.output_dir, 'registration')

config.scannet_root = '/media/yuzhiyuan/aiguo/scannet/scans'
config.shapenet_root = '/media/yuzhiyuan/aiguo/ShapeNetCore'
config.scan2cad_root = '/home/yuzhiyuan/MIRETR/scan2cad_download/scan2cad.npy'
config.scan2cad = '/home/yuzhiyuan/MIRETR/scan2cad_download/full_annotations.json'
config.process_scan2cad_root = '/home/yuzhiyuan/jittor_scan2cad_data/'


ensure_dir(config.output_dir)
ensure_dir(config.snapshot_dir)
ensure_dir(config.logs_dir)
ensure_dir(config.events_dir)
ensure_dir(config.features_dir)
ensure_dir(config.registration_dir)


# data
config.voxel_size =0.025
config.matching_radius = config.voxel_size * 2
#eval
config.re_thre = 15
config.te_thre = config.voxel_size * 4
# train config
config.train_batch_size = 1
config.train_num_worker = 10
config.train_max_num_point = 60000
config.train_use_augmentation = True
config.train_augmentation_noise = 0.005
config.train_rotation_factor = 1.0

# test config
config.test_batch_size = 1
config.test_num_worker = 0
config.test_max_num_point = 20000
config.test_tau1 = 0.1
config.test_tau2 = 0.05
config.test_registration_threshold = 0.2

# optim config
config.learning_rate = 1e-4
config.gamma = 0.95
config.momentum = 0.98
config.weight_decay = 1e-6
config.max_epoch = 60

# model - KPFCNN
config.num_layers = 4
config.in_points_dim = 3
config.first_feats_dim = 128
config.final_feats_dim = 256
config.first_subsampling_dl =config.voxel_size  #0.025   0.015 0.005 0.0015
config.in_features_dim = 1
config.conv_radius = 2.5#2.5 2.5
config.deform_radius = 5.0#5.0 2.0
config.num_kernel_points = 15
config.KP_extent = 2.0
config.KP_influence = 'linear'
config.aggregation_mode = 'sum'
config.fixed_kernel_points = 'center'
config.normalization = 'group_norm'
config.normalization_momentum = 0.02
config.deformable = False
config.modulated = False

# model - Architecture
config.architecture = ['simple', 'resnetb']
for i in range(config.num_layers - 1):
    config.architecture.append('resnetb_strided')
    config.architecture.append('resnetb')
    config.architecture.append('resnetb')
for i in range(config.num_layers - 3):
    config.architecture.append('nearest_upsample')
    config.architecture.append('unary')
config.architecture.append('nearest_upsample')
config.architecture.append('last_unary')

# model - Global
config.instance_mask_thre=0.6
config.geodesic_radis=0.15
config.cluster_thre=0.2
config.cluster_refine=True
config.ground_truth_positive_radius = config.voxel_size * 2
config.point_to_node_max_point = 64

config.sinkhorn_num_iter = 100
config.max_ref_nodes=64
config.max_neighboor=16#local transformer
config.max_sample_neighboor=16 #max_sample_neighboor<max_neighboor
config.finematch_max_point = 256#config.max_sample_neighboor*32
config.coarse_matching_num_target=128
config.coarse_matching_overlap_thresh = 0.1
config.coarse_matching_num_proposal = 128
config.coarse_matching_dual_softmax = False
config.coarse_matching_positive_overlap = 0.

config.fine_matching_max_num_corr = 15000
config.fine_matching_min_num_corr = 30
config.fine_matching_topk = 3
config.fine_matching_mutual = True
config.fine_matching_with_slack = False
config.fine_matching_confidence_threshold = 0.1
config.fine_matching_conditional_score = False
config.fine_matching_positive_radius = config.voxel_size * 2
config.fine_matching_num_registration_iter = 5

# model - Coarse level
config.coarse_tfm_feats_dim = 256
config.coarse_tfm_num_head = 4
config.coarse_tfm_architecture = ['self', 'cross','mask', 'self', 'cross', 'mask', 'self', 'cross', 'mask']
config.instance_mask_architecture=[ 'cross', 'self', 'cross', 'self', 'cross','self']
config.coarse_tfm_bin_size_d = 0.2
config.coarse_tfm_bin_size_a = 15
config.coarse_tfm_angle_k = 3

# loss - Coarse level
config.coarse_circle_loss_positive_margin = 0.1
config.coarse_circle_loss_negative_margin = 1.4
config.coarse_circle_loss_positive_optimal = 0.1
config.coarse_circle_loss_negative_optimal = 1.4
config.coarse_circle_loss_log_scale = 24#24
config.coarse_circle_loss_positive_threshold = 0.1

# loss - Fine level
config.fine_sinkhorn_loss_positive_radius = config.voxel_size * 2

# loss - Overall
config.weight_coarse_loss = 1.
config.weight_fine_loss = 1.
config.weight_mask_loss=1.
# Eval - Overall


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.link_output:
        os.symlink(config.output_dir, 'output')
