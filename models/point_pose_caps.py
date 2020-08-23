#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:44:18 2019
"""
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import sys
from pooling_point_capsule_layer import PoolingPointCapsuleLayer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from torch.autograd.gradcheck import gradgradcheck, gradcheck
import healpy as hp
USE_CUDA = True
import numpy as np
import quat_ops
import torch_autograd_solver as S


# In[Two pooling layers ]
class PoseNet(torch.nn.Module):
    def __init__(self,num_points,inter_out_channels,num_of_class, num_iterations):
        super(PoseNet, self).__init__()
        self.num_points=num_points
        self.inter_out_channels=inter_out_channels
        self.num_iterations=num_iterations
        self.num_of_class=num_of_class

        self.num_pool1_patches=256
        self.num_pool2_patches=1
        # self.num_pool4_patches=1

        self.num_neighbours2=9
        self.num_neighbours3=256
        # self.num_neighbours4=1

        self.pool1=PoolingPointCapsuleLayer(in_channels=1, out_channels=inter_out_channels,  num_iterations=num_iterations, num_neighbours=self.num_neighbours2, num_patches=self.num_pool1_patches)
        self.pool2=PoolingPointCapsuleLayer(in_channels=inter_out_channels, out_channels=num_of_class, num_iterations=num_iterations, num_neighbours=self.num_neighbours3, num_patches=self.num_pool2_patches)


    def forward(self, points_4_pool1, lrfs_4_pool1,  activation_4_pool1, points_pool1_index):
        batch_size=points_4_pool1.size(0)
        lrfs_4_pool1=lrfs_4_pool1.unsqueeze(-2)
        activation_4_pool1=activation_4_pool1.unsqueeze(-1)

    # first pooling
        pose_pool1, activation_pool1 = self.pool1(points_4_pool1, lrfs_4_pool1, activation_4_pool1)
    # generate centers and neighbours for the 2rd pooling.
        points_4_pool2=points_4_pool1[:,:,0,:].unsqueeze(1)# replace this one with the center point
        pose_4_pool2=pose_pool1.unsqueeze(1)
        activation_4_pool2=activation_pool1.unsqueeze(1)
    # second pooling
        pose_pool2, activation_pool2= self.pool2(points_4_pool2, pose_4_pool2, F.relu(activation_4_pool2))
        return pose_pool2.squeeze(), activation_pool2.squeeze()


# In[The network]
class PointCapsNet(nn.Module):
    def __init__(self, num_points, inter_out_channels, num_of_class, num_iterations):
        super(PointCapsNet, self).__init__()
        self.num_points=num_points
        self.inter_out_channels=inter_out_channels
        self.num_iterations =num_iterations
        self.num_of_class=num_of_class
        self.pose_net=PoseNet(num_points,inter_out_channels,num_of_class, num_iterations)

    def forward(self, points_pool1, lrfs_pool1,  activation_pool1, points_pool1_index):
        pose_out, a_out=self.pose_net(points_pool1, lrfs_pool1,  activation_pool1, points_pool1_index)
        return pose_out, a_out

    def one_hot(self, src, num_classes=None, dtype=None):
        num_classes = src.max().item() + 1 if num_classes is None else num_classes
        out = torch.zeros(src.size(0), num_classes, dtype=dtype)
        out.scatter_(1, src.unsqueeze(-1).expand_as(out), 1)
        return out

    def spread_loss(self, x, target, epoch):
        target = self.one_hot(target.squeeze(), x.size(1)).cuda()
#        m = min(0.1 + 0.05 * (epoch+1), 0.9)
        m= 0.1
        m = torch.tensor(m, dtype=target.dtype,device=x.device)
        act_t = (x * target).sum(dim=1)
        loss = ((F.relu(m - (act_t.view(-1, 1) - x))**2) * (1 - target))
#        loss=F.normalize(loss,p=2, dim=-1)
        loss = loss.sum(1).mean()
        return loss
    #2a2b24
