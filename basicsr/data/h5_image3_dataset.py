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
    print('Found {} h5 files in {}'.format(len(h5_file_path), file_folder_path))
    datasets = []
    for h5_file in h5_file_path:
        datasets.append(dataset(opt, h5_file))
    return ConcatDataset(datasets)


class H5ImageDataset3(data.Dataset):

    def get_frame(self, index):
        """
        Get frame at index
        @param index The index of the frame to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['images']['image{:09d}'.format(index+1)][:]

    def get_gt_frame(self, index):
        """
        Get gt frame at index
        @param index: The index of the gt frame to get
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        return self.h5_file['sharp_images']['image{:09d}'.format(index+1)][:]

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
        if self.h5_file is None:
            self.h5_file = h5py.File(self.data_path, 'r')
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0  # -1 and 1
        return xs, ys, ts, ps

    def __init__(self, opt, data_path, return_voxel=True, return_frame=True, return_gt_frame=True,
            return_mask=False, norm_voxel=True):

        super(H5ImageDataset3, self).__init__()
        # print("Image dataset3")
        # exit(0)
        self.opt = opt
        self.data_path = data_path
        self.seq_name = os.path.basename(self.data_path)
        self.seq_name = self.seq_name.split('.')[0]
        self.return_format = 'torch'

        self.return_voxel = return_voxel
        self.return_frame = return_frame
        self.return_gt_frame = opt.get('return_gt_frame', return_gt_frame)
        self.return_voxel = opt.get('return_voxel', return_voxel)
        self.return_mask = opt.get('return_mask', return_mask)
        
        self.load_h5 = opt.get('load_h5', False)
        # print(self.load_h5)
        # exit(0)
        
        self.norm_voxel = norm_voxel # -MAX~MAX -> -1 ~ 1 
        self.h5_file = None
        self.transforms={}
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.ps = self.opt['crop_size']


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
        # print(transforms_list)
        # exit(0)
        if len(transforms_list) == 0:
            self.transform = None
        elif len(transforms_list) == 1:
            self.transform = transforms_list[0]
        else:
            self.transform = Compose(transforms_list)
        # print(self.transform)
        # exit(0)
        if not self.normalize_voxels:
            self.vox_transform = self.transform

        #with h5py.File(self.data_path, 'r') as file:
        #     self.dataset_len = len(file['images'].keys())
        # exit(0)
        #print(self.dataset_len)
        # self.length = self.dataset_len - 2 # the first and the last age dont have events in both left and right
        # self.event_indices = self.compute_frame_center_indeices()
        # exit(0)
        #if self.load_h5:
        #    try:
        #        self.h5_file = h5py.File(data_path, 'r')
        #    except OSError as err:
        #        print("Couldn't open {}: {}".format(data_path, err))
        #    # if self.sensor_resolution is None:
        #    self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
        #    # print("sensor resolution = {}".format(self.sensor_resolution))
        #    self.t0 = self.h5_file['events/ts'][0]
        #    self.tk = self.h5_file['events/ts'][-1]
        #    self.num_events = self.h5_file.attrs["num_events"]
        #    self.num_frames = self.h5_file.attrs["num_imgs"]
        #    self.frame_ts = []
        #    for img_name in self.h5_file['images']:
        #        self.frame_ts.append(self.h5_file['images/{}'.format(img_name)].attrs['timestamp'])
        #    self.length = self.num_frames - 2
        #    self.event_indices = self.compute_frame_center_indeices()
        #    print("sensor resolution = {}".format(self.sensor_resolution))
        #    print(self.length)
        #else:
        #    with h5py.File(self.data_path, 'r') as file:
        #       self.length = len(file['images'].keys())
        #        # self.length = 
        #        print(self.length)
        with h5py.File(self.data_path, 'r') as file:
            self.num_frames = file.attrs["num_imgs"]
            self.num_events = file.attrs["num_events"]
            self.frame_ts = []
            for img_name in file['images']:
                self.frame_ts.append(file['images/{}'.format(img_name)].attrs['timestamp'])
            self.length = self.num_frames - 2
            self.event_indices = self.compute_frame_center_indeices(file['events/ts'])
            print(self.length)
            
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

    def __getitem__(self, index, seed=None):
        # print(index)
        if index < 0 or index >= self.__len__():
            raise IndexError
        if True:
            # index = random.randint(0, self.length-1)
            # while renew_index:
            frame = self.get_frame(index)
            # frame = self.transform_frame(frame, seed, transpose_to_CHW=True)
            frame_gt = self.get_gt_frame(index)
            # frame_gt = self.transform_frame(frame_gt, seed, transpose_to_CHW=True)
            h, w, c = frame.shape
            idx0, idx1 = self.get_event_indices(index)
            xs, ys, ts, ps = self.get_events(idx0, idx1)
            frame = self.transform_frame(frame.copy(), seed, transpose_to_CHW=True)
            frame_gt = self.transform_frame(frame_gt.copy(), seed, transpose_to_CHW=True)
            # print("h5_image3_dataset", index)
            return frame, frame_gt, ys, xs, ts, ps

            ts_0, ts_k  = ts[0], ts[-1]
            dt = ts_k-ts_0
            #seed = random.randint(0, 2 ** 32) if seed is None else seed
            # random.seed(seed)
            ix = random.randrange(0, h - self.ps + 1)
            iy = random.randrange(0, w - self.ps + 1)
            frame = frame[ix:ix+self.ps,iy:iy+self.ps,:]
            frame_gt = frame_gt[ix:ix+self.ps,iy:iy+self.ps,:]

            xs = xs - iy; ys = ys - ix


            res_index = np.where((ys >= 0) & (ys < self.ps) & (xs >=0) & (xs < self.ps))
            # print(ix, ix+self.ps, iy, iy+ self.ps)
            xs_n = np.take(ys, res_index)
            ys_n = np.take(xs, res_index)
            ts_n = np.take(ts, res_index)
            ps_n = np.take(ps, res_index)
            # print(ts_n.shape)
            #if ts_n.shape[1] > 0:
            #    break
        
        # print(frame.shape)
        if random.random() < 0.5:
            frame = frame[::-1, :, :]
            frame_gt = frame_gt[::-1, :, :]
            xs_n = self.ps-1- xs_n
        if random.random() < 0.5:
            frame = frame[:, ::-1,:]
            frame_gt = frame_gt[:, ::-1,:]
            ys_n = self.ps-1 - ys_n
        if random.random() < 0.5:
            frame = np.transpose(frame, (1,0,2))
            frame_gt = np.transpose(frame_gt, (1,0,2))
            temp = xs_n
            xs_n = ys_n
            ys_n = temp
        frame = self.transform_frame(frame.copy(), seed, transpose_to_CHW=True)
        frame_gt = self.transform_frame(frame_gt.copy(), seed, transpose_to_CHW=True)
        # else:
        
        # event_info = np.stack([ys, xs, ts, ps], axis=0)
        # print(event_info)
        
        
        # for kk in range(len(xs)):
        #xs = xs - ix
        # # ys = ys - iy
        #xs_new, ys_new, ts_new, ps_new = [], [], [], []
        #for kk in range(len(xs)):
        #    if xs[kk] >= ix and xs[kk] < ix + self.ps and ys[kk] >= iy and ys[kk] < iy+self.ps:
        #        xs_new.append(xs[kk])
        #        ys_new.append(ys[kk])
        #        ts_new.append(ts[kk])
        #        ps_new.append(ps[kk])
        #xs = torch.from_numpy(np.array(xs_new).astype(np.float32))
        #ys = torch.from_numpy(np.array(ys_new).astype(np.float32))
        #ts = torch.from_numpy(np.array(ts_new).astype(np.float32))
        # ps = torch.from_numpy(np.array(ps_new).astype(np.float32))
        #item = {}
        #item['frame'] = frame
        #item['frame_gt'] = frame_gt
        # item['voxel_x'] = xs
        
        return frame, frame_gt, xs_n, ys_n, ts_n, ps_n
        
        
        xs, ys, ts, ps = self.get_events(idx0, idx1)
        # xs, ys, ts, ps = self.preprocess_events(xs, ys, ts, ps)
        ts_0, ts_k  = ts[0], ts[-1]
        dt = ts_k-ts_0
        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        # ts = torch.from_numpy((ts-ts_0).astype(np.float32)) # ts start from 0
        ts = torch.from_numpy(ts.astype(np.float32)) # !
        ps = torch.from_numpy(ps.astype(np.float32))
        #print(xs.shape, ys.shape, ts.shape, ps.shape)
        # exit(0)
        frame = self.get_frame(index)
        frame_gt = self.get_gt_frame(index)
        frame = self.transform_frame(frame, seed, transpose_to_CHW=True) # to tensor
        frame_gt = self.transform_frame(frame_gt, seed, transpose_to_CHW=True)
        
        item={}
        #voxel = self.get_voxel(index)
        #frame = self.transform_frame(frame, seed, transpose_to_CHW=False)  # to tensor
        # normalize RGB
        #if self.mean is not None or self.std is not None:
        #    normalize(frame, self.mean, self.std, inplace=True)
        #    if self.return_gt_frame:
        #         normalize(frame_gt, self.mean, self.std, inplace=True)

        if self.return_frame:
            item['frame'] = frame
        if self.return_gt_frame:
            item['frame_gt'] = frame_gt
        if self.return_voxel:
            item['voxel'] = xs # self.transform_voxel(voxel, seed, transpose_to_CHW=False)
        #if self.return_mask:
        #    mask = self.get_mask(index)
        #    item['mask'] = self.transform_frame(mask, seed, transpose_to_CHW=False)
        item['seq'] = self.seq_name


        return item


    def __len__(self):
        return self.length

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
            #if self.transform:
            #    random.seed(seed)
            #    frame = self.transform(frame)
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
    return r