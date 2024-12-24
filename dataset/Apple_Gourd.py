from __future__ import division
import os
import numpy as np
#from scipy.ndimage import imread
from imageio import imread
import scipy.io as sio
import imageio

import torch
import torch.utils.data as data

from dataset import pms_transforms
from . import util
from descritization import conversion
from . import best_config 
np.random.seed(0)


class Apple_Gourd(data.Dataset):
    def __init__(self, args,path, split='test'):
        self.root   = os.path.join(path)
        self.split  = split
        self.args   = args
        self.objs   = util.readList(os.path.join(self.root, 'objects.txt'), sort=False)
        self.objs   = [i.strip() for i in self.objs]

        self.intens = {}
        intens_name = 'light_intensities.txt'
        print('Files for intensity: %s' % (intens_name))
        for obj in self.objs:
            self.intens[obj] = np.genfromtxt(os.path.join(self.root, obj, intens_name))
            

    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:,:,0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        if mask.dtype == bool:
            mask = mask.astype(np.float32)
            return mask
        return mask / 255.0

    def __getitem__(self, index):

        np.random.seed(index)
        obj = self.objs[index]

        light_directions = np.loadtxt(os.path.join(self.root, obj, 'light_directions.txt'))
        self.l_dir = light_directions.reshape(-1, 3)
        if self.args.benchmark == 'Light_Stage_Gallery':
            self.l_dir[:,2] = self.l_dir[:,2]*-1
        best = eval(f'best_config.best{self.args.out_img}')
        # best = [i for i in range(conversion.total_bins)]
        
        self.best_config = conversion.get_best_l(best, self.l_dir)
        
        select_idx = self.best_config
        names  = util.readList(os.path.join(self.root, obj ,'filenames.txt'), sort=False) ##
        img_list   = [os.path.join(self.root, obj, names[i]) for i in select_idx]
        intens     = [np.diag(1 / self.intens[obj][i]) for i in select_idx]
        
        
        if self.args.evaluate:
            normal_path = os.path.join(self.root, obj, 'Normal_gt.mat')
            normal = sio.loadmat(normal_path)
            normal = normal['Normal_gt']

        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            # img = np.dot(img, intens[idx])
            imgs.append(img)
        if self.args.normalize:
            imgs = pms_transforms.normalize(imgs)
        img = np.concatenate(imgs, 2)
        if self.args.normalize:
            img = img * np.sqrt(len(imgs) / self.args.train_img_num) # TODO

        mask = self._getMask(obj)
        down = 4
        if mask.shape[0] % down != 0 or mask.shape[1] % down != 0:
            pad_h = down - mask.shape[0] % down
            pad_w = down - mask.shape[1] % down
            img = np.pad(img, ((0,pad_h), (0,pad_w), (0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            mask = np.pad(mask, ((0,pad_h), (0,pad_w), (0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
            if self.args.evaluate:
                normal = np.pad(normal, ((0,pad_h), (0,pad_w), (0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        img  = img * mask.repeat(img.shape[2], 2)
        item = {'img': img, 'mask': mask}
        if self.args.evaluate:
            item['N'] = normal
        else:
            item['N'] = np.zeros_like(mask)

        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light:
            item['light'] = torch.from_numpy(self.l_dir[select_idx]).view(-1, 1, 1).float()
        item['obj'] = obj
        
        normal  = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.float32) 
        mask = self._getMask(obj)
        
        # obj_dir = os.path.join('/media/aalok/Zone_B/ashish/ReLeaPS/data/AGLS', obj)
        # if not os.path.exists(obj_dir):
        #     os.makedirs(obj_dir)
        # light_dirs = []
        # lightint = []
        # file_names = []
        # for i,s in enumerate(select_idx):
            
        #     img = imread(img_list[i])
        #     img_path = os.path.join(obj_dir, f'image_{i}.png')
        #     imageio.imwrite(img_path, img)
        #     file_names.append(f'image_{i}.png')
        #     lightint.append(self.intens[obj][s])
        #     light_dirs.append(self.l_dir[s].tolist())
        # light_dirs_path = os.path.join(obj_dir, 'light_directions.txt')
        # lightint_path = os.path.join(obj_dir, 'light_intensities.txt')
        # file_names_path = os.path.join(obj_dir, 'filenames.txt')
        
        # file_name = os.path.join('/media/aalok/Zone_B/ashish/ReLeaPS/data/AGLS', 'objects.txt')
        # if not os.path.exists(file_name):
        #     with open(file_name, 'w') as f:
        #         f.write(obj + '\n')
        # else:
        #     with open(file_name, 'a') as f:
        #         f.write(obj + '\n')
        # with open(file_names_path, 'w') as f:
        #     for name in file_names:
        #         f.write(name + '\n')
        # with open(light_dirs_path, 'w') as f:
        #     for dir in light_dirs:
        #         f.write(' '.join(map(str, dir)) + '\n')
        # with open(lightint_path, 'w') as f:
        #     for intensity in lightint:
        #         f.write(' '.join(map(str, intensity)) + '\n')
        # normal_gt_path = os.path.join(obj_dir, 'Normal_gt.mat')
        # sio.savemat(normal_gt_path, {'Normal_gt': normal})
        # # normal_gt_path = os.path.join(obj_dir, 'Normal_gt.png')
        # # imageio.imwrite(normal_gt_path, np.array(((normal+1)/2)*255,dtype=np.uint8))

        # mask_path = os.path.join(obj_dir, 'mask.png')
        # mask = (mask * 255).astype(np.uint8)
        # imageio.imwrite(mask_path, np.repeat(mask, 3, axis=2))
        
        return item

    def __len__(self):
        return len(self.objs)
