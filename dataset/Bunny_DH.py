from __future__ import division
import os
import numpy as np
#from scipy.ndimage import imread
from imageio import imread

import torch
import torch.utils.data as data

from dataset import pms_transforms
from . import util
from descritization import conversion
from . import best_config 
np.random.seed(0)

# best = best_config.best15
# best = np.random.randint(0, 48, 20)

class Bunny_DH(data.Dataset):
    def __init__(self, args, root, split='test'):
        self.root   = os.path.join(root)
        self.split  = split
        self.args   = args
        self.shape_list = util.readList(os.path.join(self.root, split + '_mtrl_bunny.txt'))

    def _getInputPath(self, index):
        shape, mtrl = self.shape_list[index].split('/')
        normal_path = os.path.join(self.root, 'Images', shape, shape + '_normal.png')
        img_dir     = os.path.join(self.root, 'Images', self.shape_list[index])
        img_list    = util.readList(os.path.join(img_dir, '%s_%s.txt' % (shape, mtrl)))

        data = np.genfromtxt(img_list, dtype='str', delimiter=' ')
        # select_idx = np.random.permutation(data.shape[0])[:self.args.in_img_num] ###
        select_idx = np.array([i for i in range(data.shape[0])])
        idxs = ['%03d' % (idx) for idx in select_idx]
        data   = data[select_idx, :]
        imgs   = [os.path.join(img_dir, img) for img in data[:, 0]]
        lights = data[:, 1:4].astype(np.float32)
        
        nn_imgs = [0 for i in range(int(conversion.total_bins))]
        nn_lights = np.zeros((int(conversion.total_bins),3))
        intens = np.genfromtxt(os.path.join(self.root, 'Images',self.shape_list[index],'gt_light_ints.txt'))
        nligth_ints = [0 for i in range(int(conversion.total_bins))]
        
        best_neigh = conversion.get_best_l([i for i in range(conversion.total_bins)], lights)
        for i in range(conversion.total_bins):
            nn_imgs[i] = imgs[best_neigh[i]]
            nn_lights[i] = lights[best_neigh[i]]
            nligth_ints[i] = np.diag(1/intens[best_neigh[i]])
        
        
        
        
        n_imgs = [0 for i in range(int(self.args.out_img))]
        n_lights = np.zeros((int(self.args.out_img),3))
        ligth_ints = [0 for i in range(int(self.args.out_img))]
        n = int(self.args.out_img)
        mat = np.zeros((3, n-1))
        for i in range(n-1):
            mat[0, i] = np.sin(i *2*np.pi / (n-1))
            mat[1, i] = np.cos(i *2*np.pi / (n-1))
            mat[2, i] = 1/np.sqrt(2)
        mat = np.sqrt(2/3) * mat
        L = np.array([[(n/(n-1))**0.5, 0, 0], [0, (n/(n-1))**0.5, 0], [0, 0, ((n-3)/(n-1))**0.5]])
        mat = np.dot(L, mat)
        mat = np.concatenate((mat, np.array([[0, 0, 1]]).T), axis=1)  
        for i in range(n):
            ind = 0
            max_cos = -1
            for j,k in enumerate(nn_lights):
                dotp = np.dot(mat[:,i], k)
                if dotp>max_cos:
                    max_cos = dotp
                    ind = j
            n_imgs[i] = nn_imgs[ind]
            n_lights[i] = nn_lights[ind]
            ligth_ints[i] = nligth_ints[ind]
        
        

        return normal_path, n_imgs, n_lights, ligth_ints

    def __getitem__(self, index):
        normal_path, img_list, lights, light_ints = self._getInputPath(index)
        normal = imread(normal_path).astype(np.float32) / 255.0 * 2 - 1
        imgs   =  []
        for a,i in enumerate(img_list):
            img = imread(i).astype(np.float32) / 255.0
            # img = np.dot(img, light_ints[a])
            imgs.append(img)
        img = np.concatenate(imgs, 2)

        h, w, c = img.shape
        # crop_h, crop_w = self.args.crop_h, self.args.crop_w
        # if self.args.rescale:
        #     sc_h = np.random.randint(crop_h, h)
        #     sc_w = np.random.randint(crop_w, w)
        #     img, normal = pms_transforms.rescale(img, normal, [sc_h, sc_w])

        # if self.args.crop:
        #     img, normal = pms_transforms.randomCrop(img, normal, [crop_h, crop_w])

        # if self.args.color_aug and not self.args.normalize:
        #     img = (img * np.random.uniform(1, 3)).clip(0, 2)

        if self.args.normalize:
            imgs = np.split(img, img.shape[2]//3, 2)
            imgs = pms_transforms.normalize(imgs)
            img = np.concatenate(imgs, 2)

        # if self.args.noise_aug:
        #     img = pms_transforms.randomNoiseAug(img, self.args.noise)

        mask   = pms_transforms.normalToMask(normal)
        normal = normal * mask.repeat(3, 2)
        norm  = np.sqrt((normal * normal).sum(2, keepdims=True))
        normal = normal / (norm + 1e-10)

        item = {'N': normal, 'img': img, 'mask': mask}
        for k in item.keys(): 
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light:
            item['light'] = torch.from_numpy(lights).view(-1, 1, 1).float()
             
        return item

    def __len__(self):
        return len(self.shape_list)
        # return 40