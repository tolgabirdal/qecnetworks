#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:37:50 2019
"""

'''
Used for 2 layers based pooling. returen 256*9 points
'''

import os
import os.path
import json
import numpy as np
import sys
#import provider
import torch
import torch.utils.data as data
from pyquaternion import Quaternion
from scipy.spatial import distance
import sys
sys.path.append('../models')
import quat_ops
import torch.nn.functional as F
import warnings
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class ModelNetDataset(data.Dataset):
    def __init__(self, root, batch_size=32, npoints=1024, split='train', normalize=False, num_of_class=10, num_gen_samples=4, class_choice=None, cache_size=100, data_aug=False, sample_pair=[0,1],rot_id=0):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        self.data_aug=data_aug
        self.num_gen_samples=num_gen_samples
        self.num_of_class=num_of_class
        self.sample_pair=sample_pair
        self.catfile = os.path.join(self.root, 'modelnet'+str(num_of_class)+'_shape_names.txt')
        self.rot_id=rot_id
#        if modelnet10:
#            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
#        else:
#            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
#        self.lrf_channel = lrf_channel

        shape_ids = {}

        shape_ids['train'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet'+str(num_of_class)+'_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet'+str(num_of_class)+'_test.txt'))]

        assert(split == 'train' or split == 'test')

        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        if class_choice== None:
            self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i],
                                                           shape_ids[split][i])+'.txt', os.path.join(self.root, shape_names[i],
                                                           shape_ids[split][i])+'.qua', os.path.join(self.root, shape_names[i],
                                                          shape_ids[split][i])+'.ds'+str(self.num_gen_samples)+'.pt', os.path.join(self.root, shape_names[i],
                                                          shape_ids[split][i])+'.idx') for i in range(len(shape_ids[split]))]

        else:
            dir_point = os.path.join(self.root, class_choice)
            self.rot_quat_file=os.path.join('/home/zhao/equcaps/3D-Group-Equ-Caps/eva/gt/', class_choice, 'pert_'+str(self.rot_id)+'.csvquat')

            fns=[]
            for file in os.listdir(dir_point):
                if file.endswith(".txt"):
                    fns.append(file)
            fns = sorted(fns)

            if split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in shape_ids['train']]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in shape_ids['test']]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            self.datapath = []
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.datapath.append(( class_choice, os.path.join(dir_point, token + '.txt'),
                                        os.path.join(dir_point, token + '.qua'),
                                        os.path.join(dir_point, token + '.ds'+str(self.num_gen_samples)+'.pt'),
                                        os.path.join(dir_point, token + '.idx')))


        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple


#    def _get_item(self, index):
    def __getitem__(self, index):
        if index in self.cache:
            point_normal_set, lrf_set, ds_index_set, wrong_ids, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
#            point_normal_set = np.loadtxt(fn[1]).astype(np.float32)
            point_normal_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

#            lrf_set = torch.load(fn[2]).float()
            lrf_set= torch.from_numpy(np.loadtxt(fn[2]).astype(np.float32))
            ds_index_set= torch.load(fn[3])

#            data_aug_rot= np.loadtxt(self.rot_quat_file, delimiter=',').astype(np.float32)
            with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
#                  data = np.loadtxt(myfile, unpack=True)
                  wrong_ids= np.loadtxt(fn[4],ndmin=1).astype(np.long)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_normal_set, lrf_set, ds_index_set, wrong_ids, cls)

        point_normal_set = torch.from_numpy(point_normal_set)
        point_set=point_normal_set[:,0:3]


        choice = np.random.choice(self.num_gen_samples, 2, replace=False)
#        choice = np.random.choice(2, 2, replace=False) # fixed for evaluation
#        choice =self.sample_pair

        points_pool2=torch.zeros(2,256,9,3)
        lrf_pool2=torch.zeros(2,256,9,4)
        activation_pool2=torch.zeros(2,256,9)
        pool2_index=torch.zeros(2,256,9)
        input_cls=torch.zeros(2,1,1).long()
#        pca_quat2=torch.zeros(2,4).long()


        point_choice = np.random.choice(len(point_set), self.npoints, replace=False)
#        point_set2048 = point_set[point_choice]
        point_set2048 = point_set.clone()

        for ds_index in range(2):
            choice_=choice[ds_index]
            ds_index_set_=ds_index_set[choice_].squeeze()

    #The index container(ds_index_set) to keep the pooling certers and neighbours, Pool1 contatiner has 1024 ceners while Pool2 has 256. The "+1" keeps the real size of the pool1 centers.
            pool1_index=ds_index_set_[0:1024]
            pool1_index0=pool1_index[0,0].clone()
            pool1_index[0,0]=-1
            pool2_index=ds_index_set_[1024:(1024+256)]
            pool2_index_=torch.clamp(pool2_index, max=1023)
            pool2_index=pool1_index[pool2_index_,0]
            pool2_index[0,0]=pool1_index0
            pool1_size=ds_index_set_[1024+256,0]

#            pca_quat2[ds_index]=ds_index_set_[1024+256,5:9]

            activation_pool2_=torch.sign(pool2_index)
            activation_pool2_=torch.clamp(activation_pool2_, min=0)

            pool2_size=len((activation_pool2_[:,0]).nonzero().squeeze())
    #        activation_pool2[pool2_size:]=0

            if wrong_ids.size !=0:
                for i in range(pool2_size):
                    for j in range(9):
                        for k in range(wrong_ids.size):
                            if(wrong_ids[k]==(pool2_index[i,j]).numpy()):
                                activation_pool2_[i,j]=0


            activation_pool2[ds_index]=activation_pool2_

            pool2_index_=pool2_index.view(-1)


            point_set[-1]=0

            if(ds_index==2 and self.data_aug):
                rotate_q=torch.randn(4)
                if(rotate_q[0]<0):
                    rotate_q=rotate_q*(-1)
                rotate_q=F.normalize(rotate_q, p=2, dim=-1)
#                rotate_q=data_aug_rot[index]
                rotate_q_=rotate_q.unsqueeze(0).expand(point_set.size(0),4)
                point_set=quat_ops.qrotv(rotate_q_, point_set)
                lrf_set=quat_ops.qmul(rotate_q_, lrf_set)
            else:
                rotate_q=torch.FloatTensor([1.0,0.0,0.0,0.0])
            points_pool2_=point_set[pool2_index_]
            points_pool2[ds_index]=points_pool2_.view(256,9,3)

            lrf_set[-1]=0
            lrf_pool2_=lrf_set[pool2_index_]
            lrf_pool2[ds_index]=lrf_pool2_.view(256,9,4)

            cls_ = torch.from_numpy(np.array([cls]).astype(np.int64))
            input_cls[ds_index]=cls_

            # activation_pool2[:,0:int(pool2_size*2/3)]=0
            # points_pool2[:,0:int(pool2_size*2/3)]=0
            # lrf_pool2[:,0:int(pool2_size*2/3)]=0
    #        return points_pool2, lrf_pool2, activation_pool2[0:256].float(), pool2_index, cls, point_set

        return points_pool2, lrf_pool2, activation_pool2, pool2_index, input_cls, point_set2048, rotate_q

    def __len__(self):
        return len(self.datapath)

    def rand_ortho_rotation_matrix(self):
        k = np.zeros((3,), dtype=int)
        k[np.random.randint(0,3)]=1 if np.random.rand()>0.5 else -1
        K = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ])
        all_theta = [0, 90, 180, 270]
        theta = np.deg2rad(all_theta[np.random.randint(0,4)])
        R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*np.dot(K,K)
        return R



if __name__ == '__main__':
#    from open3d import *
    import time
    dataset = ModelNetDataset(root='/home/zhao/dataset/modelnet40_normal_resampled/',class_choice='chair', npoints=2048, split='test', sample_pair=[0,1])

#    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    for i in range(100):
        ps = dataset[i]
