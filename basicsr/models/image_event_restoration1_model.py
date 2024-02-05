import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
import numpy as np
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')
def hash_3d(ts, ys, xs):
    return xs + 256*ys + 256*256* ts
# a * 39 + b * 255 + c
# a * 100 + b * 10 + c
# 111 =  100 
# 11 
def dehash_3d(input_s):
    ts, temp = input_s // (256*256), input_s % (256*256)
    ys, xs = (temp) // 256, temp % 256
    return ts, ys, xs

class ImageEventRestorationModel1(BaseModel):
    """Base Event-based deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageEventRestorationModel1, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.pixel_type = train_opt['pixel_opt'].pop('type')
            # print('LOSS: pixel_type:{}'.format(self.pixel_type))
            cri_pix_cls = getattr(loss_module, self.pixel_type)

            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        optim_params_lowlr = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                if k.startswith('module.offsets') or k.startswith('module.dcns'):
                    optim_params_lowlr.append(v)
                else:
                    optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}, {'params': optim_params_lowlr, 'lr': train_opt['optim_g']['lr'] * ratio}],
                                                **train_opt['optim_g'])

        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):

        self.lq = data[0]#.to(self.device)
        self.gt = data[1]#.to(self.device)
        # print(self.lq.shape, self.gt.shape)
        self.xs = data[2]#.to(self.device)
        self.ys = data[3]
        self.ts = data[4]
        self.ps = data[5]
        # 
        # print(len(self.xs), self.xs[0].shape, self.xs[0].max(), self.ys[0].max())
        #print(len(self.lq), self.lq[0].shape, self.lq[0].max(), self.lq[0].min())
        with torch.no_grad():
            self.lq = torch.stack(self.lq).cuda()
            # print(self.lq.shape)
            self.gt = torch.stack(self.gt).cuda()
            # print(self.gt.shape)
            # print(self.lq.max(), self.gt.max(), self.lq.min(), self.gt.min())
            # exit(0)
            B, C, H, W = self.lq.shape
            batch_size = len(self.xs)
            index_3d_list = []
            count_list = []
            for kk in range(batch_size):
                xs = torch.IntTensor(self.xs[kk]).cuda()
                ys = torch.IntTensor(self.ys[kk]).cuda()
                ts = torch.FloatTensor(self.ts[kk]).cuda()
                ps = torch.IntTensor(self.ps[kk]).cuda()
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
                ps = count.float().unsqueeze(0)
                zeros = torch.zeros_like(re_ts)
                index_s = torch.cat((zeros, re_ts, re_ys, re_xs), dim=0).permute(1,0).cuda()
                index_s[:,0] = kk
                # print(xs.shape, index_s.shape, count.shape)
                index_3d_list.append(index_s)
                count = count.unsqueeze(0).permute(1,0)
                # print(count.shape, count.max(), count.min())
                count_max = max(count.max(), -count.min())
                # count = count / count_max
                # exit(0)
                # print(count.shape)
                count_list.append(count)
            self.count_tensor = torch.cat(count_list, dim=0).float()
            self.index_3d_tensor = torch.cat(index_3d_list, dim=0)
            self.bs = batch_size
            self.spatial_shapes = [16, W, H]
        # print(self.count_tensor.shape, self.index_3d_tensor.shape)
        # exit(0)
            
        # lq_to_save = self.lq[0].data.numpy()
        #xs_to_save = torch.FloatTensor(self.xs[0]).data.numpy()
        #ys_to_save = torch.FloatTensor(self.ys[0]).data.numpy()
        #ts_to_save = torch.FloatTensor(self.ts[0]).data.numpy()
        #ps_to_save = torch.FloatTensor(self.ps[0]).data.numpy()
        #import os
        #if os.environ['LOCAL_RANK'] == '0':
        #np.save("image.npy", lq_to_save)
        #np.save("xs.npy", xs_to_save)
        #np.save("ys.npy", ys_to_save)
        #np.save("ts.npy", ts_to_save)
        #np.save("ps.npy", ps_to_save)
        # exit(0)
        # self.voxel=data['voxel'].to(self.device) 
        #if 'mask' in data:
        #     self.mask = data['mask'].to(self.device)
        #if 'frame_gt' in data:
        #    self.gt = data['frame_gt'].to(self.device)
        # exit(0)

    def transpose(self, t, trans_idx):
        # print('transpose jt .. ', t.size())
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return torch.rot90(t, trans_idx % 4, [2, 3])

    def transpose_inverse(self, t, trans_idx):
        # print( 'inverse transpose .. t', t.size())
        t = torch.rot90(t, 4 - trans_idx % 4, [2, 3])
        if trans_idx >= 4:
            t = torch.flip(t, [3])
        return t

    def grids_voxel(self):
        b, c, h, w = self.voxel.size()
        self.original_size_voxel = self.voxel.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)

        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.voxel[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.voxel[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_voxel = self.voxel
        self.voxel = torch.cat(parts, dim=0)
        print('----------parts voxel .. ', len(parts), self.voxel.size())
        self.idxes = idxes


    def grids(self):
        b, c, h, w = self.lq.size()  # lq is after data augment (for example, crop, if have)
        self.original_size = self.lq.size()
        assert b == 1
        crop_size = self.opt['val'].get('crop_size')
        # step_j = self.opt['val'].get('step_j', crop_size)
        # step_i = self.opt['val'].get('step_i', crop_size)
        ##adaptive step_i, step_j
        num_row = (h - 1) // crop_size + 1
        num_col = (w - 1) // crop_size + 1

        import math
        step_j = crop_size if num_col == 1 else math.ceil((w - crop_size) / (num_col - 1) - 1e-8)
        step_i = crop_size if num_row == 1 else math.ceil((h - crop_size) / (num_row - 1) - 1e-8)


        # print('step_i, stepj', step_i, step_j)
        # exit(0)


        parts = []
        idxes = []

        # cnt_idx = 0

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size >= h:
                i = h - crop_size
                last_i = True


            last_j = False
            while j < w and not last_j:
                if j + crop_size >= w:
                    j = w - crop_size
                    last_j = True
                # from i, j to i+crop_szie, j + crop_size
                # print(' trans 8')
                for trans_idx in range(self.opt['val'].get('trans_num', 1)):
                    parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                    idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})
                    # cnt_idx += 1
                j = j + step_j
            i = i + step_i
        if self.opt['val'].get('random_crop_num', 0) > 0:
            for _ in range(self.opt['val'].get('random_crop_num')):
                import random
                i = random.randint(0, h-crop_size)
                j = random.randint(0, w-crop_size)
                trans_idx = random.randint(0, self.opt['val'].get('trans_num', 1) - 1)
                parts.append(self.transpose(self.lq[:, :, i:i + crop_size, j:j + crop_size], trans_idx))
                idxes.append({'i': i, 'j': j, 'trans_idx': trans_idx})


        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        # print('parts .. ', len(parts), self.lq.size())
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size).to(self.device)
        b, c, h, w = self.original_size

        print('...', self.device)

        count_mt = torch.zeros((b, 1, h, w)).to(self.device)
        crop_size = self.opt['val'].get('crop_size')

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            trans_idx = each_idx['trans_idx']
            preds[0, :, i:i + crop_size, j:j + crop_size] += self.transpose_inverse(self.output[cnt, :, :, :].unsqueeze(0), trans_idx).squeeze(0)
            count_mt[0, 0, i:i + crop_size, j:j + crop_size] += 1.

        self.output = preds / count_mt
        self.lq = self.origin_lq
        self.voxel = self.origin_voxel
    def get_latest_images(self):
        return [self.lq[0], self.gt[0], self.output[0], self.reblurred[0]]
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        #if self.opt['datasets']['train'].get('use_mask'):
        #    preds = self.net_g(x = self.lq, event = self.voxel, mask = self.mask)

        #elif self.opt['datasets']['train'].get('return_ren'):
        #    preds = self.net_g(x = self.lq, event = self.voxel, ren = self.ren)

        #else:
        #    preds = self.net_g(x = self.lq, event = self.voxel)
        # preds = self.net_g(self.lq, self.xs, self.ys, self.ts, self.ps)
        preds, reblurred = self.net_g(self.lq, self.gt, self.count_tensor, self.index_3d_tensor, self.bs, self.spatial_shapes)
        if not isinstance(preds, list):
            preds = [preds]
        self.reblurred = reblurred
        self.output = preds[-1]
        # print(len(preds), self.output.shape, "Output")
        # exit(0)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.

            if self.pixel_type == 'PSNRATLoss':
                l_pix += self.cri_pix(*preds, self.gt)

            elif self.pixel_type == 'PSNRGateLoss':
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt, self.mask)

            elif self.pixel_type == 'PSNRLoss':
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)
            
            else:
                for pred in preds:
                    l_pix += self.cri_pix(pred, self.gt)             

            l_total += l_pix + self.cri_pix(reblurred, self.lq)
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        # if self.cri_perceptual:
        #
        #
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style

        l_total = l_total + 0 * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)
        # exit(0)
    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.size(0)  # n: batch size
            outs = []
            m = self.opt['val'].get('max_minibatch', n)  # m is the minibatch, equals to batch size or mini batch size
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n

                if self.opt['datasets']['val'].get('use_mask'):
                    pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :], mask = self.mask[i:j, :, :, :])  # mini batch all in 

                elif self.opt['datasets']['val'].get('return_ren'):
                    pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :], ren = self.ren[i:j,:])

                else:
                    pred = self.net_g(x = self.lq[i:j, :, :, :], event = self.voxel[i:j, :, :, :])  # mini batch all in 
            
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred)
                i = j

            self.output = torch.cat(outs, dim=0)  # all mini batch cat in dim0
        self.net_g.train()

    def single_image_inference(self, img, voxel, save_path):
        self.feed_data(data={'frame': img.unsqueeze(dim=0), 'voxel': voxel.unsqueeze(dim=0)})
        if self.opt['val'].get('grids') is not None:
            self.grids()
            self.grids_voxel()

        self.test()

        if self.opt['val'].get('grids') is not None:
            self.grids_inverse()
            # self.grids_inverse_voxel()

        visuals = self.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        imwrite(sr_img, save_path)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        logger = get_root_logger()
        # logger.info('Only support single GPU validation.')
        import os
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = self.opt.get('name') # !
        
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = '{:08d}'.format(cnt)

            self.feed_data(val_data)
            if self.opt['val'].get('grids') is not None:
                self.grids()
                self.grids_voxel()

            self.test()

            if self.opt['val'].get('grids') is not None:
                self.grids_inverse()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    if cnt == 1: # visualize cnt=1 image every time
                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}.png')
                        
                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                img_name,
                                                f'{img_name}_{current_iter}_gt.png')
                else:
                    print('Save path:{}'.format(self.opt['path']['visualization']))
                    print('Dataset name:{}'.format(dataset_name))
                    print('Img_name:{}'.format(img_name))
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
            cnt += 1
        pbar.close()

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
