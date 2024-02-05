import logging
import torch
from os import path as osp
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str
import numpy as np
from basicsr.models.archs.BFNet_arch import BFNet
from skimage.metrics import structural_similarity as SSIM_
from skimage.metrics import peak_signal_noise_ratio as PSNR_
from scipy.ndimage import gaussian_filter
from basicsr.utils.dist_util import get_dist_info
from basicsr.data import my_collate
import cv2, os
import spconv.pytorch as spconv
from basicsr.data.h5_image3_dataset import H5ImageDataset3

import random
import time

def create_dataloader(dataset,
                      dataset_opt,
                      num_gpu=1,
                      dist=False,
                      sampler=None,
                      seed=None):
    """Create dataloader.

    Args:
        dataset (torch.utils.data.Dataset): Dataset.
        dataset_opt (dict): Dataset options. It contains the following keys:
            phase (str): 'train' or 'val'.
            num_worker_per_gpu (int): Number of workers for each GPU.
            batch_size_per_gpu (int): Training batch size for each GPU.
        num_gpu (int): Number of GPUs. Used only in the train phase.
            Default: 1.
        dist (bool): Whether in distributed training. Used only in the train
            phase. Default: False.
        sampler (torch.utils.data.sampler): Data sampler. Default: None.
        seed (int | None): Seed. Default: None
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=my_collate,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
    elif phase in ['val', 'test']:  # validation
        dataloader_args = dict(
            dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. '
                         "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: '
                    f'num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(
            num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)

def hash_3d(ts, ys, xs):
    return xs + 1280*ys + 1280*1280* ts
def dehash_3d(input_s):
    ts, temp = input_s // (1280*1280), input_s % (1280*1280)
    ys, xs = (temp) // 1280, temp % 1280
    return ts, ys, xs
def voxel2mask(voxel):
    mask_final = np.zeros_like(voxel[0, :, :])
    mask = (voxel != 0)
    for i in range(mask.shape[0]):
        mask_final = np.logical_or(mask_final, mask[i, :, :])
    # to uint8 image
    mask_img = mask_final * np.ones_like(mask_final) * 255
    mask_img = mask_img[..., np.newaxis] # H,W,C
    mask_img = np.uint8(mask_img)

    return mask_img
def put_hot_pixels_in_voxel_(voxel, hot_pixel_range=20, hot_pixel_fraction=0.00002):
    num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
    x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
    y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
    for i in range(num_hot_pixels):
        # voxel[..., :, y[i], x[i]] = random.uniform(-hot_pixel_range, hot_pixel_range)
        voxel[..., :, y[i], x[i]] = random.randint(-hot_pixel_range, hot_pixel_range)


def add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.1):
    noise = noise_std * torch.randn_like(voxel)  # mean = 0, std = noise_std
    if noise_fraction < 1.0:
        mask = torch.rand_like(voxel) >= noise_fraction
        noise.masked_fill_(mask, 0)
    return voxel + noise
def ssim_calculate(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    # Processing input image
    img1 = np.array(img1, dtype=np.float32) / 255
    img1 = img1.transpose((2, 0, 1))

    # Processing gt image
    img2 = np.array(img2, dtype=np.float32) / 255
    img2 = img2.transpose((2, 0, 1))


    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    #make_exp_dirs(opt)
    #log_file = osp.join(opt['path']['log'],
    #                    f"test_{opt['name']}_{get_time_str()}.log")
    #logger = get_root_logger(
    #    logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    #logger.info(get_env_info())
    # logger.info(dict2str(opt))

    # create test dataset and dataloader
    #test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        dataset_opts = dataset_opt
    # exit(0)

    # create model
    # model = create_model(opt)
    model = BFNet()
    load_path = './pretrained/bfnet_g_latest.pth'
    model = model.cuda()
    load_net = torch.load(load_path)['params']
    model.load_state_dict(load_net, strict=True)
    print(load_path, "loaded ...")
    # exit(0)
    
    save_dir = "./outputs/"
    save_img = False
    if save_img and not os.path.exists(save_dir):
        os.mkdir(save_dir)
    psnr_list = []
    ssim_list = []
    time_list = []
    file_folder_path = dataset_opts['dataroot']
    if os.path.isdir(file_folder_path):
        h5_file_path = [os.path.join(file_folder_path, s) for s in os.listdir(file_folder_path)]
    for file_name in h5_file_path:
        test_set = H5ImageDataset3(dataset_opts, file_name)
        print(file_name)
        s_dir = file_name.split("/")[-1].split(".")[0]
        # exit(0)
        save_sdir = os.path.join(save_dir, s_dir)
        if save_img:
            if not os.path.exists(save_sdir):
                os.mkdir(save_sdir)
        test_loader = create_dataloader(
            test_set,
            dataset_opts,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        for idx, data in enumerate(test_loader):
            lq = data[0]
            gt = data[1]
            xs = data[2]#.to(self.device)
            ys = data[3]
            ts = data[4]
            ps = data[5]
            with torch.no_grad():
                lq = torch.stack(lq).cuda()
                gt = torch.stack(gt).cuda()
                B, C, H, W = lq.shape
                batch_size = len(xs)
                index_3d_list = []
                count_list = []
                for kk in range(batch_size):
                    # print(type(xs[kk]), xs[kk].shape, batch_size)
                    xs = torch.IntTensor(xs[kk]).cuda()
                    ys = torch.IntTensor(ys[kk]).cuda()
                    ts = torch.FloatTensor(ts[kk]).cuda()
                    ps = torch.IntTensor(ps[kk]).cuda()
                    # print(ts.shape, xs.shape)
                    ts = ts - ts.min()
                    dt = ts.max() - ts.min()
                    ts = ts / dt * 16
                    ts = ts.int().clamp(None, 15)
                    ps[ts<=7] = -ps[ts<=7]

                    pos_xs = xs[ps > 0]
                    pos_ys = ys[ps > 0]
                    pos_ts = ts[ps > 0]
                    neg_xs = xs[ps < 0]
                    neg_ys = ys[ps < 0]
                    neg_ts = ts[ps < 0]

                    pos_hashed_index = hash_3d(pos_ts, pos_ys, pos_xs)
                    #print(pos_hashed_index.shape)
                    idx_sort = torch.argsort(pos_hashed_index)
                    sorted_records_array = pos_hashed_index[idx_sort]
                    pos_vals, pos_count = torch.unique(sorted_records_array, return_counts=True)
                    # print(pos_vals.shape, pos_count.shape)
                    # print(count.max(), count.min())

                    neg_hashed_index = hash_3d(neg_ts, neg_ys, neg_xs)
                    idx_sort = torch.argsort(neg_hashed_index)
                    sorted_records_array = neg_hashed_index[idx_sort]
                    neg_vals, neg_count = torch.unique(sorted_records_array, return_counts=True)
                    neg_count = -neg_count

                    pos_vals_np = pos_vals.cpu().data.numpy()
                    neg_vals_np = neg_vals.cpu().data.numpy()
                    xy, x_ind, y_ind = np.intersect1d(pos_vals_np, neg_vals_np, return_indices=True)
                    if len(x_ind) > 0:
                        # print(pos_count[x_ind])
                        # print(neg_count[x_ind])
                        # exit(0)
                        pos_count[x_ind] += neg_count[y_ind]
                        # exit(0)
                        pos_count_np = pos_count.cpu().data.numpy()
                        neg_count_np = neg_count.cpu().data.numpy()
                        vals_np = np.concatenate((pos_vals_np, np.delete(neg_vals_np, y_ind)), axis=0)
                        count_np = np.concatenate((pos_count_np, np.delete(neg_count_np, y_ind)), axis=0)
                        #print(vals_np.shape, count_np.shape)
                        # exit(0)
                        vals = torch.IntTensor(vals_np).cuda()
                        count = torch.FloatTensor(count_np).cuda()
                        # print(vals.shape, count.shape)
                    else:
                        vals = torch.cat((pos_vals, neg_vals), dim=0)
                        count = torch.cat((pos_count, neg_count), dim=0)
                    re_ts, re_ys, re_xs = dehash_3d(vals.unsqueeze(0))
                    # ps = count.float().unsqueeze(0)
                    zeros = torch.zeros_like(re_ts)
                    index_s = torch.cat((zeros, re_ts, re_ys, re_xs), dim=0).permute(1,0).cuda()
                    index_s[:,0] = kk
                    # print(xs.shape, index_s.shape, count.shape)
                    index_3d_list.append(index_s)
                    count_max = max(count.max(), -count.min())
                    # count = count / count_max
                    count_list.append(count.unsqueeze(0).permute(1,0))
                count_tensor = torch.cat(count_list, dim=0).float()
                index_3d_tensor = torch.cat(index_3d_list, dim=0)
                bs = batch_size
                spatial_shapes = [16, W, H]
                #input_sp_tensor = spconv.SparseConvTensor(
                #    features=count_tensor,
                #    indices=index_3d_tensor.int(),
                #    spatial_shape=spatial_shapes, # [45809409, 256, 256]
                #    batch_size=bs
                #)
                # out = input_sp_tensor.dense(channels_first=True)
                # b,c1,c2,h,w = out.shape
                # voxel = out.view(b,c1*c2,h,w).permute(0,1,3,2)[0]
                #voxel[0] = voxel[0] + voxel[1] + voxel[2]
                # voxel[1] = voxel[1] + voxel[2]
                #voxel[5] = voxel[3] + voxel[4] + voxel[5]
                # voxel[4] = voxel[3] + voxel[4]
                # voxel[5] = voxel[3] + voxel[4] + voxel[5]

                # voxel = add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.05)
                # put_hot_pixels_in_voxel_(voxel, hot_pixel_range=20, hot_pixel_fraction=0.00002)
                # voxel_np = voxel.cpu().data.numpy()
                #voxel_np[0] = voxel_np[0] + voxel_np[1] + voxel_np[2]
                #voxel_np[1] = voxel_np[1] + voxel_np[2]
                #voxel_np[4] = voxel_np[3] + voxel_np[4]
                #voxel_np[5] = voxel_np[3] + voxel_np[4] + voxel_np[5]
                # print(voxel.shape, voxel_np.shape)
                # exit(0)
                #voxel = torch.FloatTensor(voxel_np).cuda().unsqueeze(0)
                # voxel_np = voxel.cpu().data.numpy()
                # mask_img = voxel2mask(voxel_np)
                #close filter
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                # mask_img_close = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel, iterations=1)
                # print(mask_img_close.shape)
                # mask_img_close = mask_img_close[np.newaxis,...] # H,W -> C,H,W  C=1
                # print(mask_img_close.shape, mask_img_close.max(), mask_img_close.min())
                # mask_img = torch.FloatTensor(mask_img_close / 255.0).cuda().unsqueeze(0)
                # exit(0)
                # print(lq.shape, voxel.shape, mask_img.shape)
                # voxel_max = max(voxel.max(), -voxel.min())
                # print(voxel_max, voxel.max(), voxel.min())
                # voxel = voxel / voxel_max
                # exit(0)
                # preds = model(lq, voxel.unsqueeze(0), mask_img)
                # print(count_tensor.shape, index_3d_tensor.shape, bs, spatial_shapes)
                # exit(0)
                t0 = time.time()
                preds, _ = model(lq, gt, count_tensor, index_3d_tensor, bs, spatial_shapes)
                torch.cuda.current_stream().synchronize()
                t1 = time.time()
                if not isinstance(preds, list):
                    preds = [preds]
                # print(preds.shape)
                output = preds[-1]
                # print(output.shape)
                # exit(0)
                # print(output.shape, gt.shape)
                # exit(0)
                lq_np = lq[0].permute(1,2,0).cpu().data.numpy() * 255
                output_np = torch.clamp(output[0].permute(1,2,0)*255, 0, 255).cpu().data.numpy()
                gt_np = gt[0].permute(1,2,0).cpu().data.numpy() * 255
                # print(output_np.max(), output_np.min(), gt_np.max(), gt_np.min())
                # exit(0)
                del lq; del gt; del count_tensor; del index_3d_tensor;
                torch.cuda.empty_cache()
                psnr = PSNR_(output_np, gt_np, data_range=255)
                ssim = ssim_calculate(output_np, gt_np)
                print(idx, psnr, ssim, t1-t0)
                # exit(0)
                if save_img:
                    # cv2.imwrite(os.path.join(save_sdir, "%03d_blur.png"%idx), lq_np)
                    cv2.imwrite(os.path.join(save_sdir, "%03d.png"%idx), output_np)
                    cv2.imwrite(os.path.join(save_sdir, "%03d_gt.png"%idx), gt_np)
                #if idx == 100:
                #     break
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                time_list.append(t1-t0)
        # exit(0)
    sum_psnr = sum(psnr_list)
    sum_ssim = sum(ssim_list)
    n_img = len(psnr_list)
    avg_time = sum(time_list[10:]) / len(time_list[10:])
    print("# Total AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(sum_psnr / n_img, sum_ssim / n_img), avg_time)
    with open("result.log", "a") as f:
        f.write(load_path + "\n")
        f.write("# Total AVG-PSNR={:.5}, AVG-SSIM={:.4} \n".format(sum_psnr / n_img, sum_ssim / n_img))
    
if __name__ == '__main__':
    main()
