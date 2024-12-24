from __future__ import division
import os
import numpy as np
#from scipy.ndimage import imread
from imageio import imread
import imageio
import scipy.io as sio

import torch
import torch.utils.data as data

from dataset import pms_transforms
from . import util
from descritization import conversion
np.random.seed(0)

class PS_SampleNet_Dataset(data.Dataset):
    def __init__(self, args, root, split='train'):
        self.root   = os.path.join(root)
        self.split  = split
        self.args   = args
        self.shape_list = util.readList(os.path.join(self.root, split + '_mtrl.txt'))
        self.bins = np.random.randint(0, conversion.total_bins, int(self.args.out_img))
        # self.bins = np.arange(0, conversion.total_bins, 1)
        

    def _getInputPath(self, index):
        shape, mtrl = self.shape_list[index].split('/')
        normal_path = os.path.join(self.root, 'Images', shape, shape + '_normal.png')
        img_dir     = os.path.join(self.root, 'Images', self.shape_list[index])
        img_list    = util.readList(os.path.join(img_dir, '%s_%s.txt' % (shape, mtrl)))

        data = np.genfromtxt(img_list, dtype='str', delimiter=' ')
        # select_idx = np.random.permutation(data.shape[0])[:self.args.in_img_num] ###
        select_idx = np.array([i for i in range(64)])
        idxs = ['%03d' % (idx) for idx in select_idx]
        data   = data[select_idx, :]
        imgs   = [os.path.join(img_dir, img) for img in data[:, 0]]
        lights = data[:, 1:4].astype(np.float32)
        if self.args.selection == 'random':
            n_imgs = [0 for i in range(int(self.args.out_img))]
            n_lights = np.zeros((int(self.args.out_img),3))
            for i,j in enumerate(self.bins):
                tid = conversion.get_best_neigh(j,lights)
                n_imgs[i] = imgs[tid]
                n_lights[i] = lights[tid]
        else:
            n_imgs = [0 for i in range(conversion.total_bins)]
            n_lights = np.zeros((conversion.total_bins,3))
            for i in range(conversion.total_bins):
                tid = conversion.get_best_neigh(i,lights)
                n_imgs[i] = imgs[tid]
                n_lights[i] = lights[tid]

        return normal_path, n_imgs, n_lights

    def __getitem__(self, index):
        normal_path, img_list, lights = self._getInputPath(index)
        normal = imread(normal_path).astype(np.float32) / 255.0 * 2 - 1
        imgs   =  []
        img_dict = {}
        for i in img_list:
            if i in img_dict:
                img = img_dict[i]
            else:
                img = imread(i).astype(np.float32) / 255.0
                img_dict[i] = img
            imgs.append(img)
        img = np.concatenate(imgs, 2)

        h, w, c = img.shape
        crop_h, crop_w = self.args.crop_h, self.args.crop_w
        if self.args.rescale:
            sc_h = np.random.randint(crop_h, h)
            sc_w = np.random.randint(crop_w, w)
            img, normal = pms_transforms.rescale(img, normal, [sc_h, sc_w])

        if self.args.crop:
            img, normal = pms_transforms.randomCrop(img, normal, [crop_h, crop_w])

        if self.args.color_aug and not self.args.normalize:
            img = (img * np.random.uniform(1, 3)).clip(0, 2)

        if self.args.normalize:
            imgs = np.split(img, img.shape[2]//3, 2)
            imgs = pms_transforms.normalize(imgs)
            img = np.concatenate(imgs, 2)

        if self.args.noise_aug:
            img = pms_transforms.randomNoiseAug(img, self.args.noise)

        mask   = pms_transforms.normalToMask(normal)
        normal = normal * mask.repeat(3, 2)
        norm  = np.sqrt((normal * normal).sum(2, keepdims=True))
        normal = normal / (norm + 1e-10)

        item = {'N': normal, 'img': img, 'mask': mask}
        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light:
            item['light'] = torch.from_numpy(lights).view(-1, 1, 1).float()
            
            
        # Create directory named 'obj' if it doesn't exist
        # obj = self.shape_list[index].split('/')[-2]+'_'+self.shape_list[index].split('/')[-1]
        # obj_dir = os.path.join('/media/aalok/Zone_B/ashish/PS-FCN/data/Blobby/pmsData', obj)

        # if not os.path.exists(obj_dir):
        #     os.makedirs(obj_dir)
            
        # txt_file = os.path.join('/media/aalok/Zone_B/ashish/PS-FCN/data/Blobby/pmsData', 'objects.txt')
        # if not os.path.exists(txt_file):
        #     with open(txt_file, 'w') as f:
        #         f.write(obj + '\n')
        # else:
        #     with open(txt_file, 'a') as f:
        #         f.write(obj + '\n')
        # light_dirs = lights
        # lightint = lights
        # file_names = []
        # for i,s in enumerate(img_list):
            
        #     img = imread(img_list[i])
        #     img_path = os.path.join(obj_dir, f'image_{i}.png')
        #     imageio.imwrite(img_path, img)
        #     file_names.append(f'image_{i}.png')
            
        # light_dirs_path = os.path.join(obj_dir, 'light_directions.txt')
        # lightint_path = os.path.join(obj_dir, 'light_intensities.txt')
        # file_names_path = os.path.join(obj_dir, 'filenames.txt')
        # with open(file_names_path, 'w') as f:
        #     for name in file_names:
        #         f.write(name + '\n')
        # with open(light_dirs_path, 'w') as f:
        #     for dir in light_dirs:
        #         f.write(' '.join(map(str, dir)) + '\n')
        # with open(lightint_path, 'w') as f:
        #     for intensity in lightint:
        #         f.write(' '.join(map(str, intensity)) + '\n')
        # # Store Normal_gt.mat
        # normal_gt_path = os.path.join(obj_dir, 'Normal_gt.mat')
        # sio.savemat(normal_gt_path, {'Normal_gt': normal})

        # mask_path = os.path.join(obj_dir, 'mask.png')
        # mask = (mask * 255).astype(np.uint8)
        # imageio.imwrite(mask_path, np.repeat(mask, 3, axis=2))
        return item

    def __len__(self):
        return len(self.shape_list)
        # return 40
