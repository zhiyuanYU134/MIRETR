import torch.utils.data
from IPython import embed


from vision3d.datasets.registration.dataset_kpconv import get_dataloader,ROBISENCEDataset


def ROBI_train_data_loader(engine, config):
    train_dataset = ROBISENCEDataset(config.voxel_size,config.robi_root, 'train',config.matching_radius,
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
    valid_dataset = ROBISENCEDataset(config.voxel_size,config.robi_root,'val',config.matching_radius,
                                                 max_point=config.train_max_num_point,
                                                 use_augmentation=False,
                                                 augmentation_noise=config.train_augmentation_noise,
                                                 rotation_factor=config.train_rotation_factor)
    
    valid_sampler = torch.utils.data.DistributedSampler(valid_dataset) if engine.distributed else None
    valid_data_loader, _ = get_dataloader(valid_dataset, config, config.test_batch_size, config.test_num_worker,
                                          shuffle=False,
                                          sampler=valid_sampler,
                                          neighborhood_limits=neighborhood_limits,
                                          drop_last=False)


    return train_data_loader, valid_data_loader,neighborhood_limits
def ROBI_test_data_loader(engine, config):
    test_dataset = ROBISENCEDataset(config.voxel_size,config.robi_root, 'test',config.matching_radius,
                                                 max_point=config.train_max_num_point,
                                                 use_augmentation=config.train_use_augmentation,
                                                 augmentation_noise=config.train_augmentation_noise,
                                                 rotation_factor=config.train_rotation_factor)
    train_sampler = torch.utils.data.DistributedSampler(test_dataset) if engine.distributed else None
    test_data_loader, neighborhood_limits = get_dataloader(test_dataset, config,
                                                            config.train_batch_size,
                                                            config.train_num_worker,
                                                            shuffle=True,
                                                            sampler=train_sampler,
                                                            neighborhood_limits=None,
                                                            drop_last=True)
    return test_data_loader, neighborhood_limits



def main():
    from config import config




if __name__ == '__main__':
    main()
