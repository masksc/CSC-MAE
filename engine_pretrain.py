# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import math
import sys
import pandas as pd
from typing import Iterable

import cv2
import torch
import torch.nn as nn
import torchvision


import util.misc as misc
import util.lr_sched as lr_sched

import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw, Image

from train_sc import cluster_embedding


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

    for data_iter_step, (samples, _, features) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _, _ = model(samples, features, mask_ratio=args.mask_ratio)

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


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    embeddings = []
    targets = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        sample = batch[0]
        target = batch[1]
        features = batch[2]
        sample = sample.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            embedding = model(sample, features, mask_ratio=0.)[-1]
        
        embeddings.append(embedding)
        targets.append(target)

    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)
    embeddings = concat_all_gather(embeddings)[:,1:].mean(dim=1)
    targets = concat_all_gather(targets)

    embeddings = embeddings.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    print('embeddings:', embeddings.shape)
    print('targets:', targets.shape)

    # save pred and features
    df = pd.DataFrame(columns = ["dataset", "pred", "features"])
    df.loc[df.shape[0]] = [args.dataset, targets, embeddings]
    df.to_pickle(f"./output/pickle_results/real_data/real_data_cscmae.pkl")

    res_eval = cluster_embedding(embeddings, data_loader.dataset.cluster_number, targets, save_pred = args.save_pred,
                                 leiden_n_neighbors=args.leiden_n_neighbors, cluster_methods = args.cluster_methods)

    return res_eval


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def evaluate_vis_attn(data_loader, model, device, args):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch_iter, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = batch[0]
        target = batch[1]
        features = batch[2]
        samples = samples.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            attentions_batch = model(samples, features, mask_ratio=0., return_attn=True)[-1]
        
        # add visualizations
        # attentions_batch = model(images, return_type='attention')
        attentions_batch = torch.stack(attentions_batch)[-1]
        print('attentions:', attentions_batch.shape)          # [B, 12, 197, 197]

        # make the image divisible by the patch size
        # w, h = images.shape[1] - images.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        # img = img[:, :w, :h].unsqueeze(0)

        all_samples_list = []
        # samples = (normalize(samples)*255).long()

        for j in range(len(attentions_batch)):

            image = samples[j].unsqueeze(0).detach().cpu()

            w_featmap = image.shape[-2] // 4
            h_featmap = image.shape[-1] // 4

            nh = attentions_batch.shape[1] # number of head

            # we keep only the output patch attention
            # attentions = attentions_batch.sum(dim=-2)[j, :, :].reshape(nh, -1).detach().cpu()
            attentions = attentions_batch[j, :, 0, 1:].reshape(nh, -1).detach().cpu()

            attentions = attentions.reshape(nh, w_featmap, h_featmap)

            attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=4, mode="nearest")[0].cpu().numpy()
            
            print('attentions:', attentions.shape)                   # [12, 224, 224]
            
            # save attentions heatmaps
            # os.makedirs(args.output_dir, exist_ok=True)
            # denorm_image = torchvision.utils.make_grid(image, normalize=True, scale_each=True)
            print('denorm_image:', image.shape)             # [1, 3, 224, 224]
            # save raw images
            # all_attentions_list = [image]

            # add mean head attention
            with io.BytesIO() as buffer:
                plt.imsave(buffer, attentions.mean(0), format = "png")
                buffer.seek(0)
                attn = Image.open(buffer)
                from PIL import ImageFont
                attn = attn.convert("RGBA")
                # draw = ImageDraw.Draw(new_img)
                
                # add label
                # font = ImageFont.FreeTypeFont('fonts/arial.ttf', size=16)
                # # draw.text((224//2, 20), text, anchor="ms", align="center", font=font)

                # text_size = font.getsize(text)
                # button_size = (text_size[0]+20, text_size[1]+20)
                # button_img = Image.new('RGBA', button_size, (25, 25, 25, 160))
                # button_draw = ImageDraw.Draw(button_img)
                # button_draw.text((10, 10), text, font=font)

                # new_img.paste(button_img, (224//2-button_size[0]//2, 10))
                # new_img = new_img.convert("RGB")

                attn_array = np.asarray(attn)
            attention_map = torch.from_numpy(attn_array)[:,:,:3].permute(2, 0, 1).unsqueeze(0)
            print('attention_map:', attention_map.shape)
            # all_attentions_list.append(attention_map)

            # # add max head attention
            # with io.BytesIO() as buffer:
            #     plt.imsave(buffer, attentions.max(0), format = "png")
            #     buffer.seek(0)
            #     attn = Image.open(buffer)
            #     attn_array = np.asarray(attn)
            # attention_map = torch.from_numpy(attn_array)[:,:,:3].permute(2, 0, 1).unsqueeze(0)
            # print('attention_map:', attention_map.shape)
            # all_attentions_list.append(attention_map)

            # # add min head attention
            # with io.BytesIO() as buffer:
            #     plt.imsave(buffer, attentions.min(0), format = "png")
            #     buffer.seek(0)
            #     attn = Image.open(buffer)
            #     attn_array = np.asarray(attn)
            # attention_map = torch.from_numpy(attn_array)[:,:,:3].permute(2, 0, 1).unsqueeze(0)
            # print('attention_map:', attention_map.shape)
            # all_attentions_list.append(attention_map)

            # for h in range(nh):
            #     attention = attentions[h]
            #     # plt.imshow(attention)
            #     with io.BytesIO() as buffer:
            #         plt.imsave(buffer, attention, format = "png")
            #         buffer.seek(0)
            #         attn = Image.open(buffer)
            #         attn_array = np.asarray(attn)
            #     attention_map = torch.from_numpy(attn_array)[:,:,:3].permute(2, 0, 1).unsqueeze(0)
            #     print('attention_map:', attention_map.shape)

            #     all_attentions_list.append(attention_map)

            # all_samples = torch.cat(all_attentions_list, dim=0)
            all_samples = attention_map
            all_samples_list.append(all_samples)                

        all_samples_tensor = torch.cat(all_samples_list, dim=0)
        k_grid_img = torchvision.utils.make_grid(all_samples_tensor, nrow=8, padding=5, pad_value=255)
        k_grid_img = k_grid_img.permute(1, 2, 0).numpy()[:,:,::-1]

        os.makedirs('./attn_images_only_padding20', exist_ok=True)
        cv2.imwrite(f'./attn_images_only_padding20/{batch_iter}_attn_head.png', k_grid_img)


        if batch_iter == 50:
            break