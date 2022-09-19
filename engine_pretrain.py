# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
from random import sample
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import torchvision.transforms.functional as TF


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        
        # 可视化gt和 pred
        if log_writer is not None and (data_iter_step + 1) % args.vis_interval == 0:
            N, num_patches, _ = pred.size()
            _, C, H, W = samples.size()
            sample_patches = model.patchify(samples) 

            recon_vis = model.unpatchify(
                pred * mask.reshape(N, num_patches, 1) 
                + sample_patches * (1 - mask.reshape(N, num_patches, 1))
            )  # mask 0-kept 1-removed

            masked_vis = model.unpatchify(
                sample_patches * (1-mask.reshape(N, num_patches, -1))
            ) 

            cmp = torch.stack([masked_vis ,recon_vis, samples], dim=1)
            cmp = cmp.permute(0,2,3,1,4).reshape(N,C,H,-1)
            # de-normalized
            cmp = TF.normalize(cmp, mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225])
            cmp = TF.normalize(cmp, mean = [ -0.485, -0.456, -0.406 ],  std = [ 1., 1., 1. ])
            cmp = torch.clip(cmp, min=0, max=1)
            
            log_writer.add_images('train/recon_vis', cmp, data_iter_step + 1)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}