import torch.utils.data
from IPython import embed
import numpy as np
import open3d as o3d

from vision3d.datasets.registration.dataset_kpconv import Process_Scan2cadKPConvDataset,Scan2cadKPConvDataset,get_dataloader
from vision3d.utils.open3d_utils import make_open3d_colors, make_open3d_point_cloud


def Scan2cad_train_data_loader(engine, config):
    train_dataset = Scan2cadKPConvDataset(config.voxel_size,config.scannet_root, config.shapenet_root, config.scan2cad_root,'train',config.matching_radius,
                                                 max_point=config.train_max_num_point,
                                                 use_augmentation=config.train_use_augmentation,
                                                 augmentation_noise=config.train_augmentation_noise,
                                                 rotation_factor=config.train_rotation_factor)
    train_sampler = torch.utils.data.DistributedSampler(train_dataset) if engine.distributed else None
    train_data_loader, neighborhood_limits = get_dataloader(train_dataset, config,
                                                            config.train_batch_size,
                                                            config.train_num_worker,
                                                            shuffle=False,
                                                            sampler=train_sampler,
                                                            neighborhood_limits=None,
                                                            drop_last=True)
    valid_dataset = Scan2cadKPConvDataset(config.voxel_size,config.scannet_root, config.shapenet_root, config.scan2cad_root,'val',config.matching_radius,
                                                 max_point=config.train_max_num_point,
                                                 use_augmentation=config.train_use_augmentation,
                                                 augmentation_noise=config.train_augmentation_noise,
                                                 rotation_factor=config.train_rotation_factor)
    
    valid_sampler = torch.utils.data.DistributedSampler(valid_dataset) if engine.distributed else None
    valid_data_loader, _ = get_dataloader(valid_dataset, config, config.test_batch_size, config.test_num_worker,
                                          shuffle=False,
                                          sampler=valid_sampler,
                                          neighborhood_limits=neighborhood_limits,
                                          drop_last=False)
    return train_data_loader,valid_data_loader, neighborhood_limits

def Scan2cad_test_data_loader(engine, config):
    train_dataset = Scan2cadKPConvDataset(config.voxel_size,config.scannet_root, config.shapenet_root, config.scan2cad_root,'train',config.matching_radius,
                                                 max_point=config.train_max_num_point,
                                                 use_augmentation=config.train_use_augmentation,
                                                 augmentation_noise=config.train_augmentation_noise,
                                                 rotation_factor=config.train_rotation_factor)
    train_sampler = torch.utils.data.DistributedSampler(train_dataset) if engine.distributed else None
    train_data_loader, neighborhood_limits = get_dataloader(train_dataset, config,
                                                            config.train_batch_size,
                                                            config.train_num_worker,
                                                            shuffle=False,
                                                            sampler=train_sampler,
                                                            neighborhood_limits=None,
                                                            drop_last=True)
    valid_dataset = Scan2cadKPConvDataset(config.voxel_size,config.scannet_root, config.shapenet_root, config.scan2cad_root,'test',config.matching_radius,
                                                 max_point=config.train_max_num_point,
                                                 use_augmentation=config.train_use_augmentation,
                                                 augmentation_noise=config.train_augmentation_noise,
                                                 rotation_factor=config.train_rotation_factor)
    
    valid_sampler = torch.utils.data.DistributedSampler(valid_dataset) if engine.distributed else None
    valid_data_loader, _ = get_dataloader(valid_dataset, config, config.test_batch_size, config.test_num_worker,
                                          shuffle=False,
                                          sampler=valid_sampler,
                                          neighborhood_limits=neighborhood_limits,
                                          drop_last=False)
    return valid_data_loader


def Process_Scan2cad_test_data_loader(engine, config):
    train_dataset = Process_Scan2cadKPConvDataset(config.process_scan2cad_root,'train',
                                                 max_point=config.train_max_num_point,
                                                 use_augmentation=config.train_use_augmentation,
                                                 augmentation_noise=config.train_augmentation_noise,
                                                 rotation_factor=config.train_rotation_factor)
    train_sampler = torch.utils.data.DistributedSampler(train_dataset) if engine.distributed else None
    train_data_loader, neighborhood_limits = get_dataloader(train_dataset, config,
                                                            config.train_batch_size,
                                                            config.train_num_worker,
                                                            shuffle=False,
                                                            sampler=train_sampler,
                                                            neighborhood_limits=None,
                                                            drop_last=True)
    valid_dataset = Process_Scan2cadKPConvDataset(config.process_scan2cad_root,'test',
                                                 max_point=config.train_max_num_point,
                                                 use_augmentation=config.train_use_augmentation,
                                                 augmentation_noise=config.train_augmentation_noise,
                                                 rotation_factor=config.train_rotation_factor)
    
    valid_sampler = torch.utils.data.DistributedSampler(valid_dataset) if engine.distributed else None
    valid_data_loader, _ = get_dataloader(valid_dataset, config, config.test_batch_size, config.test_num_worker,
                                          shuffle=False,
                                          sampler=valid_sampler,
                                          neighborhood_limits=neighborhood_limits,
                                          drop_last=False)
    return valid_data_loader
def _get_voxel_lines(config):
    voxel_size = config.first_subsampling_dl
    for block in config.architecture:
        if 'strided' in block or 'pool' in block:
            voxel_size *= 2

    base_vertices = np.asarray(
        [[-voxel_size, -voxel_size, -voxel_size],
         [-voxel_size, -voxel_size, voxel_size],
         [-voxel_size, voxel_size, -voxel_size],
         [-voxel_size, voxel_size, voxel_size],
         [voxel_size, -voxel_size, -voxel_size],
         [voxel_size, -voxel_size, voxel_size],
         [voxel_size, voxel_size, -voxel_size],
         [voxel_size, voxel_size, voxel_size]]
    )
    base_vertices *= 0.5

    base_lines = np.asarray(
        [[0, 1],
         [0, 2],
         [0, 4],
         [1, 3],
         [1, 5],
         [2, 3],
         [2, 6],
         [3, 7],
         [4, 5],
         [4, 6],
         [5, 7],
         [6, 7]],
        dtype=np.int64
    )

    return base_vertices, base_lines, voxel_size


def draw_voxelized_points(points_c, points_f, base_vertices, base_lines, voxel_size):
    min_corner = np.amin(points_f, axis=0, keepdims=True)
    original_corner = np.floor(min_corner / voxel_size) * voxel_size
    voxel_centers = np.floor((points_c - original_corner) / voxel_size) * voxel_size + original_corner + voxel_size / 2
    vertices = []
    lines = []
    for i in range(voxel_centers.shape[0]):
        vertices.append(voxel_centers[i] + base_vertices)
        lines.append(base_lines + i * 8)
    vertices = np.concatenate(vertices, axis=0)
    lines = np.concatenate(lines, axis=0)
    print(lines.shape[0] // 8)
    line_colors = np.ones((lines.shape[0], 1)) * np.asarray([[1, 0, 1]])
    cubes = o3d.geometry.LineSet()
    cubes.points = o3d.utility.Vector3dVector(vertices)
    cubes.lines = o3d.utility.Vector2iVector(lines)
    cubes.colors = o3d.utility.Vector3dVector(line_colors)

    colors_f = make_open3d_colors(points_f, [0, 0, 1], scaling=True)
    pcd_f = make_open3d_point_cloud(points_f, colors=colors_f)

    colors_c = make_open3d_colors(points_c, [1, 0, 0])
    pcd_c = make_open3d_point_cloud(points_c, colors=colors_c)

    # o3d.visualization.draw_geometries([pcd_f, pcd_c])
    # o3d.visualization.draw_geometries([pcd_f, pcd_c, cubes])
    o3d.visualization.draw_geometries([pcd_c, cubes])


def main():
    from config import config



if __name__ == '__main__':
    main()
