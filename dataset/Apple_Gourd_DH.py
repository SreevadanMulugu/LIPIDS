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


class Apple_Gourd_DH(data.Dataset):
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
        # best = eval(f'best_config.best{self.args.out_img}')
        best = [i for i in range(conversion.total_bins)]
        
        self.best_config = conversion.get_best_l(best, self.l_dir)
        
        select_idx = self.best_config
        names  = util.readList(os.path.join(self.root, obj ,'filenames.txt'), sort=False) ##
        img_list   = [os.path.join(self.root, obj, names[i]) for i in select_idx]
        intens     = [np.diag(1 / self.intens[obj][i]) for i in select_idx]
        
        n_imgs = [0 for i in range(int(self.args.out_img))]
        n_lights = np.zeros((int(self.args.out_img),3))
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
            ind2 = 0
            max_cos = -1
            for j,k in enumerate(select_idx):
                dotp = np.dot(mat[:,i], self.l_dir[k])
                if dotp>max_cos:
                    max_cos = dotp
                    ind = j
                    ind2 = k
            n_imgs[i] = img_list[ind]
            n_lights[i] = self.l_dir[ind2]
        
        
        if self.args.evaluate:
            normal_path = os.path.join(self.root, obj, 'Normal_gt.mat')
            normal = sio.loadmat(normal_path)
            normal = normal['Normal_gt']

        imgs = []
        for idx, img_name in enumerate(n_imgs):
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
            item['light'] = torch.from_numpy(n_lights).view(-1, 1, 1).float()
        item['obj'] = obj
        
        normal  = np.zeros((1001,1001,3),dtype=np.float32) 
        mask = self._getMask(obj)
        obj_dir = os.path.join('/media/aalok/Zone_B/ashish/SDMUniPS/my_data', obj+'.data')
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
        light_dirs = []
        lightint = []
        file_names = []
        for i,s in enumerate(select_idx):
            
            img = imread(img_list[i])
            img_path = os.path.join(obj_dir, f'L ({i}).png')
            imageio.imwrite(img_path, img)
            file_names.append(f'image_{s}.png')
            lightint.append(self.intens[obj][s])
            light_dirs.append(self.l_dir[s].tolist())
        light_dirs_path = os.path.join(obj_dir, 'light_directions.txt')
        lightint_path = os.path.join(obj_dir, 'light_intensities.txt')
        file_names_path = os.path.join(obj_dir, 'filenames.txt')
        
        file_name = os.path.join('/media/aalok/Zone_B/ashish/SDMUniPS/my_data', 'objects.txt')
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
        # Store Normal_gt.mat
        # normal_gt_path = os.path.join(obj_dir, 'Normal_gt.mat')
        # sio.savemat(normal_gt_path, {'Normal_gt': normal})
        normal_gt_path = os.path.join(obj_dir, 'Normal_gt.png')
        imageio.imwrite(normal_gt_path, np.array(((normal+1)/2)*255,dtype=np.uint8))

        mask_path = os.path.join(obj_dir, 'mask.png')
        mask = (mask * 255).astype(np.uint8)
        imageio.imwrite(mask_path, np.repeat(mask, 3, axis=2))
        return item

    def __len__(self):
        return len(self.objs)
