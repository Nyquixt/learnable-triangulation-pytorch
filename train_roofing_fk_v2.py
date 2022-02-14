import os
import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle

import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn.models.fk import VolumetricAngleRegressor
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, HeatmapMSELoss

from mvn.utils import misc, cfg
from mvn.datasets import roofing_fk
from mvn.datasets import utils_fk as dataset_utils
from mvn.utils.skeleton import Skeleton

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")

    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--logdir", type=str, default="/Vol1/dbstore/datasets/k.iskakov/logs/multi-view-net-repr", help="Path, where logs will be stored")

    args = parser.parse_args()
    return args


def setup_human36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = roofing_fk.RoofingMultiViewDataset(
            roofing_root=config.dataset.train.roofing_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.labels_path,
            scale_bbox=config.dataset.train.scale_bbox,
            kind=config.kind,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    # val
    val_dataset = roofing_fk.RoofingMultiViewDataset(
        roofing_root=config.dataset.val.roofing_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.labels_path,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, train_sampler


def setup_dataloaders(config, is_train=True, distributed_train=False):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler = setup_human36m_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader, train_sampler


def setup_experiment(config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title

    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))

    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer


def one_epoch(model, skeleton, kpt_criterion, rot_criterion, opt, config, dataloader, device, epoch, n_iters_total=0, is_train=True, master=False, experiment_dir=None, writer=None):
    name = "train" if is_train else "val"

    if is_train:
        model.train()
    else:
        model.eval()

    metric_dict = defaultdict(list)

    results = defaultdict(list)

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        end = time.time()

        iterator = enumerate(dataloader)
        if is_train and config.opt.n_iters_per_epoch is not None:
            iterator = islice(iterator, config.opt.n_iters_per_epoch)

        for iter_i, batch in iterator:
            # measure data loading time
            data_time = time.time() - end

            if batch is None:
                print("Found None batch")
                continue

            images_batch, keypoints_3d_batch_gt, keypoints_3d_validity_batch_gt, _, rotations_gt, proj_matricies_batch = dataset_utils.prepare_batch(batch, device)

            keypoints_pred, rotations_pred = model(images_batch, proj_matricies_batch, skeleton, batch)

            # calculate loss
            total_loss = 0.0
            loss1 = kpt_criterion(keypoints_pred, keypoints_3d_batch_gt, keypoints_3d_validity_batch_gt)
            loss2 = rot_criterion(rotations_pred, rotations_gt)
            loss = loss1 + loss2
            total_loss += loss
            metric_dict['total_loss'].append(total_loss.item())

            if is_train:
                opt.zero_grad()
                total_loss.backward()

                if hasattr(config.opt, "grad_clip"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.opt.grad_clip / config.opt.lr)

                # metric_dict['grad_norm_times_lr'].append(config.opt.lr * misc.calc_gradient_norm(filter(lambda x: x[1].requires_grad, model.named_parameters())))

                opt.step()

            # save answers for evaluation
            if not is_train:
                results['keypoints_pred'].append(keypoints_pred.detach().cpu().numpy())
                results['rotations_pred'].append(rotations_pred.detach().cpu().numpy())
                results['indexes'].append(batch['indexes'])

            # dump to tensorboard per-iter loss/metric stats
            if is_train:
                for title, value in metric_dict.items():
                    writer.add_scalar(f"{name}/{title}", value[-1], n_iters_total)

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            # dump to tensorboard per-iter time stats
            writer.add_scalar(f"{name}/batch_time", batch_time, n_iters_total)
            writer.add_scalar(f"{name}/data_time", data_time, n_iters_total)

            n_iters_total += 1

    # calculate evaluation metrics
    if master:
        if not is_train:
            results['keypoints_pred'] = np.concatenate(results['keypoints_pred'], axis=0)
            results['rotations_pred'] = np.concatenate(results['rotations_pred'], axis=0)
            results['indexes'] = np.concatenate(results['indexes'])

            try:
                scalar_metric_kpt, full_metric_kpt = dataloader.dataset.evaluate(results['keypoints_pred'])
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                scalar_metric_kpt, full_metric_kpt = 0.0, {}

            try:
                scalar_metric_rot, full_metric_rot = dataloader.dataset.evaluate_angles(results['rotations_pred'])
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                scalar_metric_rot, full_metric_rot = 0.0, {}

            metric_dict['dataset_metric_kpt'].append(scalar_metric_kpt)
            metric_dict['dataset_metric_rot'].append(scalar_metric_rot)

            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

            # dump results
            with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                pickle.dump(results, fout)

            # dump full metric kpt
            with open(os.path.join(checkpoint_dir, "metric_kpt.json".format(epoch)), 'w') as fout:
                json.dump(full_metric_kpt, fout, indent=4, sort_keys=True)

            # dump full metric rot
            with open(os.path.join(checkpoint_dir, "metric_rot.json".format(epoch)), 'w') as fout:
                json.dump(full_metric_rot, fout, indent=4, sort_keys=True)

        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)
    
    return n_iters_total


def init_distributed(args):
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        return False

    torch.cuda.set_device(args.local_rank)

    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"

    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    return True


def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    is_distributed = init_distributed(args)
    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0

    if is_distributed:
        device = torch.device(args.local_rank)
    else:
        device = torch.device(0)

    # config
    config = cfg.load_config(args.config)
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    model = VolumetricAngleRegressor(config, device=device).to(device)

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint)
        for key in list(state_dict.keys()):
            if 'volume_net.back_layers' in key:
                state_dict.pop(key)
                continue
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded pretrained weights for whole model")

    # criterion
    if config.opt.criterion == "MSE":
        kpt_criterion = KeypointsMSELoss()
    elif config.opt.criterion == "MSESmooth":
        kpt_criterion = KeypointsMSESmoothLoss()
    elif config.opt.criterion == "MAE":
        kpt_criterion = KeypointsMAELoss()

    rot_criterion = nn.MSELoss()

    # freeze pretrained weights of backbone
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze backbone final layer
    for param in model.backbone.final_layer.parameters():
        param.requires_grad = True
    # unfreeze process_feature layer
    for param in model.process_features.parameters():
        param.requires_grad = True
    # unfreeze v2v
    for param in model.volume_net.parameters():
        param.requires_grad = True

    # optimizer
    opt = None
    if not args.eval:
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)

    lr = config.opt.lr

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master:
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)

    # multi-gpu
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])

    # initialize skeleton
    skeleton = Skeleton(
        offsets=[[   0.,            0.,            0.        ],
                [ -72.61139782,  -67.88703531,   85.75745005],
                [   0.,         -431.96588571,    0.        ],
                [   0.,         -458.09642169,    0.        ],
                [ -72.61139782,  -67.88703531,  -85.75745005],
                [   0.,         -431.96588571,    0.        ],
                [   0.,         -483.62178433,    0.        ],
                [-103.42245772,   83.70337939,    0.        ],
                [   3.24029647,  381.54362506,  174.59600608],
                [  11.6987705,  -254.79626644,   -8.53999566],
                [  -8.72358411, -233.87277473,   13.49641693],
                [   3.24029647,  381.54362506, -174.59600608],
                [  11.40787689, -248.46067721,    8.32764598],
                [  -8.53615171, -228.84785207,  -13.20643682]],
        parents=[-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12],
        joints_left=[4, 5, 6, 11, 12, 13],
        joints_right = [1, 2, 3, 8, 9, 10],
        device=device
    )

    if not args.eval:
        # train loop
        n_iters_total_train, n_iters_total_val = 0, 0
        for epoch in range(config.opt.n_epochs):
            lr = adjust_learning_rate(opt, epoch, lr, config.opt.schedule, config.opt.gamma)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            n_iters_total_train = one_epoch(model, skeleton, kpt_criterion, rot_criterion, opt, config, train_dataloader, device, epoch, n_iters_total=n_iters_total_train, is_train=True, master=master, experiment_dir=experiment_dir, writer=writer)
            n_iters_total_val = one_epoch(model, skeleton, kpt_criterion, rot_criterion, opt, config, val_dataloader, device, epoch, n_iters_total=n_iters_total_val, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

            if master:
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))

            print(f"{n_iters_total_train} iters done.")
    else:
        if args.eval_dataset == 'train':
            one_epoch(model, skeleton, kpt_criterion, rot_criterion, opt, config, train_dataloader, device, 0, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)
        else:
            one_epoch(model, skeleton, kpt_criterion, rot_criterion, opt, config, val_dataloader, device, 0, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    main(args)