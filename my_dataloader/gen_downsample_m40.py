#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 01:05:23 2019
"""

'''
   Generate random samples from 10k to 2K/ Genreate uniform samples from 2k to 512 to 64.
'''

import os
import os.path
import json
import numpy as np
import sys
import torch
import torch.utils.data as data
from pyquaternion import Quaternion
from scipy.spatial import distance
import sys
sys.path.append('../models')
import quat_ops
import torch.nn.functional as F

def pc_normalize(pc):
   # l = pc.shape[0]
   centroid = np.mean(pc, axis=0)
   pc = pc - centroid
   m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
   pc = pc / m
   return pc


class ModelNetDataset(data.Dataset):
   def __init__(self, root, batch_size=32, npoints=1024, split='train', normalize=False,  num_of_class=10, lrf_channel=True, num_gen_samples=1, cache_size=150, data_aug=True):
       self.root = root
       self.batch_size = batch_size
       self.npoints = npoints
       self.normalize = normalize
       self.data_aug=data_aug
       self.num_gen_samples=num_gen_samples

       self.catfile = os.path.join(self.root, 'modelnet'+str(num_of_class)+'_shape_names.txt')
       self.cat = [line.rstrip() for line in open(self.catfile)]
       self.classes = dict(zip(self.cat, range(len(self.cat))))
       self.lrf_channel = lrf_channel

       shape_ids = {}

       shape_ids['train'] = [line.rstrip() for line in open(
               os.path.join(self.root, 'modelnet'+str(num_of_class)+'_train.txt'))]
       shape_ids['test'] = [line.rstrip() for line in open(
               os.path.join(self.root, 'modelnet'+str(num_of_class)+'_test.txt'))]

       assert(split == 'train' or split == 'test')
       shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
       # list of (shape_name, shape_txt_file_path) tuple
       self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i],
                                                      shape_ids[split][i])+'.txt', os.path.join(self.root, shape_names[i],
                                                      shape_ids[split][i])+'.ds'+str(self.num_gen_samples)+'.pt') for i in range(len(shape_ids[split]))]

       self.cache_size = cache_size  # how many data points to cache in memory
       self.cache = {}  # from index to (point_set, cls) tuple



#    def _get_item(self, index):
   def __getitem__(self, index):
#         exists = os.path.isfile(fn[2])
#         if not exists:
#             return 'ds file exists'
       fn = self.datapath[index]
       cls = self.classes[self.datapath[index][0]]
       cls = np.array([cls]).astype(np.int32)
#            point_normal_set = np.loadtxt(fn[1]).astype(np.float32)
       point_normal_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
       #lrf_set = torch.load(fn[2])
     #  if len(self.cache) < self.cache_size:
    #       self.cache[index] = (point_normal_set, cls)

        #Init the container to keep the pooling certers and neighbours, Pool1 contatiner has 1024 ceners while Pool2 has 256. The "+1" keeps the real size of the pool1 centers.
       all_smaples_index=torch.empty(self.num_gen_samples,1024+256+1,9).long()


       for sample_no in range(self.num_gen_samples):
           choice = np.random.choice(len(point_normal_set), self.npoints, replace=False)
           point_set2048 = point_normal_set[choice, 0:3]
           point_set2048=torch.from_numpy(point_set2048).float()

           if(self.data_aug):
                rotate_q=torch.randn(4)
                if(rotate_q[0]<0):
                    rotate_q=rotate_q*(-1)
                rotate_q=F.normalize(rotate_q, p=2, dim=-1)
                rotate_q=rotate_q.unsqueeze(0).expand(2048,4)
                point_set2048=quat_ops.qrotv(rotate_q, point_set2048)

           choice=torch.from_numpy(choice).long()
#           point_set2048=point_normal_set2048[:,0:3]

           # downsmapling:
           distance_matrix = distance.cdist(point_set2048, point_set2048, 'euclidean')

           diam=torch.sqrt((torch.max(point_set2048[:,0])-torch.min(point_set2048[:,0]))**2 + (torch.max(point_set2048[:,1])-torch.min(point_set2048[:,1]))**2 + (torch.max(point_set2048[:,2])-torch.min(point_set2048[:,2]))**2)
           tau_p=300


           num_of_chosen_points_pool1=0
           tau = tau_p * diam / 10000
#           tau_s=tau**2
           points_pool1_index=torch.zeros(2048,9).long()
           pool1_index_4_save=torch.zeros(2048,9).long()
           points_neighbour_counts1=torch.zeros(2048).long()
           for point_index in range(len(point_set2048)):
               j = 0
               found = False
               while (j < num_of_chosen_points_pool1 and found==False):
                   index_compare=points_pool1_index[j,0]
                   if ( distance_matrix[point_index,index_compare]< tau ):
                       found = True # criteria satisfied, we should not add.
                       if(points_neighbour_counts1[j]<8):
                           points_pool1_index[j,points_neighbour_counts1[j]+1]=point_index
                           pool1_index_4_save[j,points_neighbour_counts1[j]+1]=choice[point_index]
                           points_neighbour_counts1[j]+=1
                   j = j+1
               if (found==False): # sample the point iff no points around are found
                   points_pool1_index[num_of_chosen_points_pool1,0]=point_index
                   pool1_index_4_save[num_of_chosen_points_pool1,0]=choice[point_index]
                   num_of_chosen_points_pool1+=1

           points_pool1_index_=points_pool1_index[0:num_of_chosen_points_pool1]
           pool1_index_4_save[num_of_chosen_points_pool1:]=-1
           num_of_chosen_points_pool2=0
           tau_p=tau_p*2
           tau = tau_p * diam / 10000
#           tau_s=tau**2


           points_pool2_index=torch.zeros(256,9).long()
           pool2_index_4_save=torch.zeros(256,9).long()
           points_neighbour_counts2=torch.zeros(256).long()
           for point_index in range(num_of_chosen_points_pool1):
#               activation_pool1[point_index,0:points_neighbour_counts1[point_index]]=1
               j = 0
               found = False
               while (j < num_of_chosen_points_pool2 and found==False):
                   point_index_=points_pool1_index_[point_index,0]
                   index_compare=points_pool1_index_[points_pool2_index[j,0],0]
                   if (distance_matrix[point_index_,index_compare]<tau):
                       found = True # criteria satisfied, we should not add.
                       if(points_neighbour_counts2[j]<8):
                           points_pool2_index[j,points_neighbour_counts2[j]+1]=point_index
                           pool2_index_4_save[j,points_neighbour_counts2[j]+1]=point_index
                           points_neighbour_counts2[j]+=1
                   j = j+1
               if (found==False): # sample the point iff no points around are found
                   points_pool2_index[num_of_chosen_points_pool2,0]=point_index
                   pool2_index_4_save[num_of_chosen_points_pool2,0]=point_index
                   num_of_chosen_points_pool2+=1
                   if(num_of_chosen_points_pool2>255):
                       break

           points_pool2_index[num_of_chosen_points_pool2:]=-1
#           pool2_index_4_save[num_of_chosen_points_pool2:]=-1
           pool2_index_4_save[num_of_chosen_points_pool2:]=1023
           pool1_size=num_of_chosen_points_pool1*torch.ones([1,9]).long()
           pool1_index_4_save=pool1_index_4_save[0:1024]

           all_smaples_index[sample_no]=torch.cat((pool1_index_4_save,pool2_index_4_save,pool1_size),dim=0)
#           print('num_of_chosen_points_pool2',num_of_chosen_points_pool2)
       torch.save(all_smaples_index,fn[2])
#        print(fn[2])
       return fn[2]

   def __len__(self):
       return len(self.datapath)

   def num_channel(self):
       if self.lrf_channel:
           return 7
       else:
           return 3

   def calc_distances(self,p0, points):
       return ((p0 - points)**2).sum()




if __name__ == '__main__':
#    from open3d import *
   import time
   dataset = ModelNetDataset(root='/home/zhao/dataset/modelnet40_normal_resampled/', npoints=2048, split='test', num_of_class=40, num_gen_samples=5,data_aug=False)
   loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=14)
   for fn in loader:
#        print(pool1_index.shape())
       print(fn)
       print('\n')
