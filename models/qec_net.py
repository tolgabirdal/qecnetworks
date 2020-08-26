#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import quat_ops
import torch.nn.functional as F
import torch_autograd_solver as S
from qec_module import QecModule


# In[Two pooling layers ]
class QecNet(torch.nn.Module):
    def __init__(self,num_points,inter_out_channels,num_of_class, num_iterations):
        super(QecNet, self).__init__()
        self.num_points=num_points
        self.inter_out_channels=inter_out_channels
        self.num_iterations=num_iterations
        self.num_of_class=num_of_class

        self.num_pool1_patches=256
        self.num_pool2_patches=1

        self.num_neighbours2=9
        self.num_neighbours3=256

        self.pool1=QecModule(in_channels=1, out_channels=inter_out_channels,  num_iterations=num_iterations, num_neighbours=self.num_neighbours2, num_patches=self.num_pool1_patches)
        self.pool2=QecModule(in_channels=inter_out_channels, out_channels=num_of_class, num_iterations=num_iterations, num_neighbours=self.num_neighbours3, num_patches=self.num_pool2_patches)


    def forward(self, points4pool1, lrfs4pool1,  activation4pool1, pointspool1index):
        batch_size=points4pool1.size(0)
        lrfs4pool1=lrfs4pool1.unsqueeze(-2)
        activation4pool1=activation4pool1.unsqueeze(-1)

    # first pooling
        pose_pool1, activation_pool1 = self.pool1(points4pool1, lrfs4pool1, activation4pool1)
    # generate centers and neighbours for the 2rd pooling.
        points4pool2=points4pool1[:,:,0,:].unsqueeze(1)# replace this one with the center point
        pose4pool2=pose_pool1.unsqueeze(1)
        activation4pool2=activation_pool1.unsqueeze(1)
    # second pooling
        pose_pool2, activation_pool2= self.pool2(points4pool2, pose4pool2, F.relu(activation4pool2))
        return pose_pool2.squeeze(), activation_pool2.squeeze()


    def one_hot(self, src, num_classes=None, dtype=None):
        num_classes = src.max().item() + 1 if num_classes is None else num_classes
        out = torch.zeros(src.size(0), num_classes, dtype=dtype)
        out.scatter_(1, src.unsqueeze(-1).expand_as(out), 1)
        return out

    def spread_loss(self, x, target, epoch):
        target = self.one_hot(target.squeeze(), x.size(1)).cuda()
        m = torch.tensor(0.1, dtype=target.dtype,device=x.device)
        act_t = (x * target).sum(dim=1)
        loss = ((F.relu(m - (act_t.view(-1, 1) - x))**2) * (1 - target))
        loss = loss.sum(1).mean()
        return loss

    def pose_diff_loss(self, pose):
        temp=torch.clamp(torch.abs((pose[0] * pose[1]).sum(dim=-1)),max=0.9999)           
        distance = 2*torch.acos(temp)/np.pi
        return distance.mean()