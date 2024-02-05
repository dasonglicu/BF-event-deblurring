# ------------------------------------------------------------------------
# Modified from (https://github.com/TimoStoff/events_contrast_maximization)
# ------------------------------------------------------------------------
from torch.utils import data as data
import pandas as pd
from torchvision.transforms.functional import normalize
from tqdm import tqdm
import os
from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, random_augmentation
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
from torch.utils.data.dataloader import default_collate
import h5py
# local modules
from basicsr.data.h5_augment import *
from torch.utils.data import ConcatDataset


"""
    Data augmentation functions.
    modified from https://github.com/TimoStoff/events_contrast_maximization

    @InProceedings{Stoffregen19cvpr,
    author = {Stoffregen, Timo and Kleeman, Lindsay},
    title = {Event Cameras, Contrast Maximization and Reward Functions: An Analysis},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
    } 
"""


def concatenate_h5_datasets(dataset, opt):
    """
    file_path: path that contains the h5 file
    """
    file_folder_path = opt['dataroot']
    

    if os.path.isdir(file_folder_path):
        h5_file_path = [os.path.join(file_folder_path, s) for s in os.listdir(file_folder_path)]
    elif os.path.isfile(file_folder_path):
        h5_file_path = pd.read_csv(file_folder_path, header=None).values.flatten().tolist()
    else:
        raise Exception('{} must be data_file.txt or base/folder'.format(file_folder_path))
    print('h5_image1, Found {} h5 files in {}'.format(len(h5_file_path), file_folder_path))
    datasets = []
    # print(h5_file_path)
    # file_folder_path2 = "/workspace/EFNet/datasets/GOPRO_rawevents/test"
    #file_folder_path2 = '/workspace/EFNet/datasets/REBlur_rawevents/test'
    # h5_file_path2 = [os.path.join(file_folder_path2, s) for s in os.listdir(file_folder_path2)]
    # print(h5_file_path2)
    # exit(0)
    # h5_file_path = [os.path.join(file_folder_path, "GOPR0372_07_00.h5")]
    # print(h5_folder_path)
    # exit(0)
    for kkk in range(len(h5_file_path)):    
    #for h5_file in h5_file_path:
        datasets.append(dataset(opt, h5_file_path[kkk], h5_file_path[kkk]))
    return ConcatDataset(datasets)


class H5ImageDataset(data.Dataset):

    def get_frame(self, index):
        """
        Get frame at index
        @param index The index of the frame to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['images']['image{:09d}'.format(index)][:]

    def get_gt_frame(self, index):
        """
        Get gt frame at index
        @param index: The index of the gt frame to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['sharp_images']['image{:09d}'.format(index)][:]

    def get_voxel(self, index):
        """
        Get voxels at index
        @param index The index of the voxels to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['voxels']['voxel{:09d}'.format(index)][:]

    def get_mask(self, index):
        """
        Get event mask at index
        @param index The index of the event mask to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['masks']['mask{:09d}'.format(index)][:]
    def get_events(self, idx0, idx1):
        if self.h5_file2 is None:
            self.h5_file2 = h5py.File(self.data_path2, 'r')
        xs = self.h5_file2['events/xs'][idx0:idx1]
        ys = self.h5_file2['events/ys'][idx0:idx1]
        ts = self.h5_file2['events/ts'][idx0:idx1]
        ps = self.h5_file2['events/ps'][idx0:idx1] * 2.0 - 1.0  # -1 and 1
        return xs, ys, ts, ps


    def __init__(self, opt, data_path, data_path2, return_voxel=True, return_frame=True, return_gt_frame=True,
            return_mask=False, norm_voxel=True):

        super(H5ImageDataset, self).__init__()
        print("LALALA Image dataset1")
        exit(0)
        self.opt = opt
        self.data_path = data_path
        self.data_path2 = data_path2
        self.seq_name = os.path.basename(self.data_path)
        self.seq_name = self.seq_name.split('.')[0]
        self.return_format = 'torch'

        self.return_voxel = return_voxel
        self.return_frame = return_frame
        self.return_gt_frame = opt.get('return_gt_frame', return_gt_frame)
        self.return_voxel = opt.get('return_voxel', return_voxel)
        self.return_mask = opt.get('return_mask', return_mask)
        
        self.norm_voxel = norm_voxel # -MAX~MAX -> -1 ~ 1 
        self.h5_file = None
        self.h5_file2 = None
        self.transforms={}
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None


        if self.opt['norm_voxel'] is not None:
            self.norm_voxel = self.opt['norm_voxel']   # -MAX~MAX -> -1 ~ 1 
        
        if self.opt['return_voxel'] is not None:
            self.return_voxel = self.opt['return_voxel']

        if self.opt['crop_size'] is not None:
            self.transforms["RandomCrop"] = {"size": self.opt['crop_size']}
        
        if self.opt['use_flip']:
            self.transforms["RandomFlip"] = {}

        if 'LegacyNorm' in self.transforms.keys() and 'RobustNorm' in self.transforms.keys():
            raise Exception('Cannot specify both LegacyNorm and RobustNorm')

        self.normalize_voxels = False
        for norm in ['RobustNorm', 'LegacyNorm']:
            if norm in self.transforms.keys():
                vox_transforms_list = [eval(t)(**kwargs) for t, kwargs in self.transforms.items()]
                del (self.transforms[norm])
                self.normalize_voxels = True
                self.vox_transform = Compose(vox_transforms_list)
                break

        transforms_list = [eval(t)(**kwargs) for t, kwargs in self.transforms.items()]

        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)

        if not self.normalize_voxels:
            self.vox_transform = self.transform

        with h5py.File(self.data_path, 'r') as file:
            self.dataset_len = len(file['images'].keys())
        with h5py.File(self.data_path2, 'r') as file:
            self.num_frames = file.attrs["num_imgs"]
            self.num_events = file.attrs["num_events"]
            self.frame_ts = []
            for img_name in file['images']:
                self.frame_ts.append(file['images/{}'.format(img_name)].attrs['timestamp'])
            self.length = self.num_frames - 2
            self.event_indices = self.compute_frame_center_indeices(file['events/ts'])
            self.sensor_resolution = file.attrs['sensor_resolution'][0:2]
            print(self.length, len(transforms_list), self.mean, self.std, self.sensor_resolution)
        # exit(0)
    def find_ts_index(self, file, timestamp):
        idx = binary_search_h5_dset(file, timestamp)
        return idx
    def compute_frame_center_indeices(self, file):
        """
        For each frame, find the start and end indices of the events around the
        frame, the start and the end are at the middle between the frame and the 
        neighborhood frames
        """
        frame_indices = []
        start_idx = self.find_ts_index(file, (self.frame_ts[0]+self.frame_ts[1])/2)
        for i in range(1, len(self.frame_ts)-1): 
            end_idx = self.find_ts_index(file, (self.frame_ts[i]+self.frame_ts[i+1])/2)
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices
    def get_event_indices(self, index):
        """
        Get start and end indices of events at index
        @param Desired data index
        @returns Start and end indices of events at index
        """
        idx0, idx1 = self.event_indices[index]
        if not (idx0 >= 0 and idx1 <= self.num_events):
            raise Exception("WARNING: Event indices {},{} out of bounds 0,{}".format(idx0, idx1, self.num_events))
        return int(idx0), int(idx1)
    def get_events_accumulate_voxel_frame_center(self, xs, ys, ts, ps):
        """
        Given events, return events accumulate voxel with frame centered
        The num_bins have to be even!
        @param xs tensor containg x coords of events
        @param ys tensor containg y coords of events
        @param ts tensor containg t coords of events
        @param ps tensor containg p coords of events
        @returns Voxel grid of input events
        """
        voxel_grid = events_to_accumulate_voxel_torch(xs, ys, ts, ps, 6, sensor_size=self.sensor_resolution, keep_middle=False)
        # voxel_grid = events_to_accumulate_voxel_torch_2(xs, ys, ts, ps, self.num_bins, sensor_size=self.sensor_resolution)
        
        return voxel_grid

    def __getitem__(self, index, seed=None):

        if index < 0 or index >= self.__len__():
            raise IndexError
        seed = random.randint(0, 2 ** 32) if seed is None else seed
        item={}
        frame = self.get_frame(index)
        h, w, c = frame.shape
        idx0, idx1 = self.get_event_indices(index)
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        # ts = torch.from_numpy((ts-ts_0).astype(np.float32)) # ts start from 0
        ts = torch.from_numpy(ts.astype(np.float32)) # !
        # ps = torch.abs(torch.from_numpy(ps.astype(np.float32)))
        ps = torch.from_numpy(ps.astype(np.float32))
        voxel2 = self.get_events_accumulate_voxel_frame_center(xs, ys, ts, ps)
        # voxel2 = voxel2.data.numpy()
        # voxel = self.transform_voxel(voxel, seed)
        
        
        # print(frame.shape, xs.shape, ys.shape, ts.shape, ps.shape)
        # exit(0)
        if self.return_gt_frame:
            frame_gt = self.get_gt_frame(index)
            frame_gt = self.transform_frame(frame_gt, seed, transpose_to_CHW=False)
        
        voxel = self.get_voxel(index)
        frame = self.transform_frame(frame, seed, transpose_to_CHW=False)  # to tensor
        # print(voxel.shape, voxel2.shape)
        # exit(0)
        # normalize RGB
        if self.mean is not None or self.std is not None:
            normalize(frame, self.mean, self.std, inplace=True)
            if self.return_gt_frame:
                normalize(frame_gt, self.mean, self.std, inplace=True)

        if self.return_frame:
            item['frame'] = frame
        if self.return_gt_frame:
            item['frame_gt'] = frame_gt
        if self.return_voxel:
            item['voxel'] = self.transform_voxel(voxel, seed, transpose_to_CHW=False)
            item['voxel2'] = voxel2 # self.transform_voxel(voxel2, seed, transpose_to_CHW=False)
        if self.return_mask:
            mask = self.get_mask(index)
            item['mask'] = self.transform_frame(mask, seed, transpose_to_CHW=False)
            
        item['seq'] = self.seq_name
        item['xs'] = ys
        item['ys'] = xs
        item['ts'] = ts
        item['ps'] = ps
        return item


    def __len__(self):
        return self.dataset_len

    def transform_frame(self, frame, seed, transpose_to_CHW=False):
        """
        Augment frame and turn into tensor
        @param frame Input frame
        @param seed  Seed for random number generation
        @returns Augmented frame
        """
        if self.return_format == "torch":
            if transpose_to_CHW:
                frame = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255  # H,W,C -> C,H,W

            else:
                frame = torch.from_numpy(frame).float() / 255 # 0-1
            if self.transform:
                random.seed(seed)
                frame = self.transform(frame)
        return frame

    def transform_voxel(self, voxel, seed, transpose_to_CHW):
        """
        Augment voxel and turn into tensor
        @param voxel Input voxel
        @param seed  Seed for random number generation
        @returns Augmented voxel
        """
        if self.return_format == "torch":
            if transpose_to_CHW:
                voxel = torch.from_numpy(voxel.transpose(2, 0, 1)).float()# H,W,C -> C,H,W

            else:
                if self.norm_voxel:
                    voxel = torch.from_numpy(voxel).float() / abs(max(voxel.min(), voxel.max(), key=abs))  # -1 ~ 1
                else:
                    voxel = torch.from_numpy(voxel).float()

            if self.vox_transform:
                random.seed(seed)
                voxel = self.vox_transform(voxel)
        return voxel


    @staticmethod
    def collate_fn(data, event_keys=['events'], idx_keys=['events_batch_indices']):
        """
        Custom collate function for pyTorch batching to allow batching events
        """
        collated_events = {}
        events_arr = []
        end_idx = 0
        batch_end_indices = []
        for idx, item in enumerate(data):
            for k, v in item.items():
                if not k in collated_events.keys():
                    collated_events[k] = []
                if k in event_keys:
                    end_idx += v.shape[0]
                    events_arr.append(v)
                    batch_end_indices.append(end_idx)
                else:
                    collated_events[k].append(v)
        for k in collated_events.keys():
            try:
                i = event_keys.index(k)
                events = torch.cat(events_arr, dim=0)
                collated_events[event_keys[i]] = events
                collated_events[idx_keys[i]] = batch_end_indices
            except:
                collated_events[k] = default_collate(collated_events[k])
        return collated_events

def binary_search_h5_dset(dset, x, l=None, r=None, side='left'):
    """
    Binary search for a timestamp in an HDF5 event file, without
    loading the entire file into RAM
    @param dset The HDF5 dataset
    @param x The timestamp being searched for
    @param l Starting guess for the left side (0 if None is chosen)
    @param r Starting guess for the right side (-1 if None is chosen)
    @param side Which side to take final result for if exact match is not found
    @returns Index of nearest event to 'x'
    """
    l = 0 if l is None else l
    r = len(dset)-1 if r is None else r
    while l <= r:
        mid = l + (r - l)//2;
        midval = dset[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return 
def events_to_accumulate_voxel_torch(xs, ys, ts, ps, B, device=None, sensor_size=(180, 240), keep_middle=True):
    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0]
    t_mid = ts[0] + (dt/2)
    
    # left of the mid -
    tend = t_mid
    end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend)
    for bi in range(int(B/2)):
        tstart = ts[0] + (dt/B)*bi
        beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart)
        vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end], device, sensor_size=sensor_size,
                clip_out_of_range=False)
        bins.append(-vb) # !
    # self
    #if keep_middle:
    #    bins.append(torch.zeros_like(vb))  # TODO!!!
    # right of the mid +
    tstart = t_mid
    beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart)
    for bi in range(int(B/2), B):
        tend = ts[0] + (dt/B)*(bi+1)
        end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend)
        vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end], device, sensor_size=sensor_size,
                clip_out_of_range=False)
        bins.append(vb)

    bins = torch.stack(bins)
    return bins
def binary_search_torch_tensor(t, l, r, x, side='left'):
    """
    Binary search implemented for pytorch tensors (no native implementation exists)
    @param t The tensor
    @param x The value being searched for
    @param l Starting lower bound (0 if None is chosen)
    @param r Starting upper bound (-1 if None is chosen)
    @param side Which side to take final result for if exact match is not found
    @returns Index of nearest event to 'x'
    """
    if r is None:
        r = len(t)-1
    while l <= r:
        mid = l + (r - l)//2;
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r
def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True, default=0):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
    @param xs Tensor of x coords of events
    @param ys Tensor of y coords of events
    @param ps Tensor of event polarities/weights
    @param device The device on which the image is. If none, set to events device
    @param sensor_size The size of the image sensor/output image
    @param clip_out_of_range If the events go beyond the desired image size,
       clip the events to fit into the image
    @param interpolation Which interpolation to use. Options=None,'bilinear'
    @param padding If bilinear interpolation, allow padding the image by 1 to allow events to fit:
    @returns Event image from the events
    """
    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = (torch.ones(img_size)*default).to(device)
    if interpolation == 'bilinear' and xs.dtype is not torch.long and xs.dtype is not torch.long:
        pxs = (xs.floor()).float()
        pys = (ys.floor()).float()
        dxs = (xs-pxs).float()
        dys = (ys-pys).float()
        pxs = (pxs*mask).long()
        pys = (pys*mask).long()
        masked_ps = ps.squeeze()*mask
        interpolate_to_image(pxs, pys, dxs, dys, masked_ps, img)
    else:
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        try:
            mask = mask.long().to(device)
            xs, ys = xs*mask, ys*mask
            img.index_put_((ys, xs), ps, accumulate=True)
            # print("able to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
            #     ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
        except Exception as e:
            print("Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
                ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
            raise e
    return img
