import os.path as osp
import argparse
import time
import torch
import torch.utils.data
from config import config
from vision3d.engine import Engine
from vision3d.utils.torch_utils import to_cuda

from model import create_model
from dataset import ROBI_test_data_loader
from loss import Evaluator
from vision3d.utils.point_cloud_utils import  apply_transform



eps = 1e-8

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', type=int, required=True, default=0)


def run_one_epoch(
        engine,
        epoch,
        data_loader,
        model,
        evaluator,
        training=True
):
    if training:
        model.train()
        if engine.distributed:
            data_loader.sampler.set_epoch(epoch)
    else:
        model.eval()
    num_iter_per_epoch = len(data_loader)
    inlier_ratio_num=0
    mean_pre=0
    mean_recall=0
    count=0
    for i, data_dict in enumerate(data_loader):
        print(i)
        data_dict = to_cuda(data_dict)
        with torch.no_grad():
            output_dict = model(data_dict)
            estimated_transforms_gt=data_dict['transform']
            all_ref_corr_points = output_dict['all_ref_corr_points']
            all_src_corr_points = output_dict['all_src_corr_points']
            result_dict = evaluator(output_dict, data_dict)
            mean_pre+=result_dict['precision']
            mean_recall+=result_dict['recall']
            print('precision',100*mean_pre/(count+1))
            print('recall',100*mean_recall/(count+1))    
            corr_tensor=torch.cat((all_src_corr_points, all_ref_corr_points), dim=1)
            align_src_points = apply_transform(corr_tensor[:,:3].unsqueeze(0), estimated_transforms_gt)
            rmse = torch.linalg.norm(align_src_points - corr_tensor[:,3:].unsqueeze(0), dim=-1)<(0.005)
            inlier_ratio=(rmse.float().sum(0)>0)
            inlier_ratio=inlier_ratio.float().sum()/len(inlier_ratio)
            if len(all_src_corr_points)>1:
                inlier_ratio_num+=inlier_ratio
            print('IR',100*inlier_ratio_num/(count+1)) 
            count+=1            
    print('precision',100*mean_pre/count)
    print('recall',100*mean_recall/count)    
    recall=100*mean_recall/count
    precision=100*mean_pre/count
    print("f1",2 * (precision ) * (recall) / ((recall  +precision)))

    message = 'precision: {:.3f}, '.format(100*mean_pre/count) + \
                            'recall: {:.3f}, '.format(100*mean_recall/count) + \
                                'inlier_ratio: {:.3f}, '.format(100*inlier_ratio_num/(count))
    engine.logger.critical(message)



def main():
    parser = make_parser()
    log_file = osp.join(config.logs_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        engine.args.test_epoch=39

        start_time = time.time()

        test_loader,neighborhood_limits = ROBI_test_data_loader(engine, config)
        loading_time = time.time() - start_time

        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)

        model = create_model(config).cuda()
        evaluator = Evaluator(config).cuda()

        engine.register_state(model=model)

        snapshot = osp.join(config.snapshot_dir, 'epoch-{}.pth.tar'.format(engine.args.test_epoch))
        engine.load_snapshot(snapshot)

        start_time = time.time()
        run_one_epoch(
                engine, engine.args.test_epoch, test_loader, model, evaluator, training=False
            )

        loading_time = time.time() - start_time
        message = ' test_one_epoch: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)


if __name__ == '__main__':
    main()
