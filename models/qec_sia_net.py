#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import quat_ops
import torch.nn.functional as F
import torch_autograd_solver as S
from qec_module import QecModule


class QecSiaNet(torch.nn.Module):
    def __init__(self,num_points,inter_out_channels,num_of_class, num_iterations):
        super(QecSiaNet, self).__init__()
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


    def forward(self, points4pool1_, lrfs4pool1_,  activation4pool1_, pointspool1index_):
        pose_out_sia=[]
        a_out_sia=[]
        for i in range(2):
            points4pool1=points4pool1_[:,i]
            lrfs4pool1=lrfs4pool1_[:,i]
            activation4pool1=activation4pool1_[:,i]
            pointspool1index=pointspool1index_[:,i]
 
            ## same as the single forward                      
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
            
            pose_out_sia.append(pose_pool2.squeeze())
            a_out_sia.append(activation_pool2.squeeze())
        
        pose_out_sia=torch.stack(pose_out_sia,0)
        a_out_sia=torch.stack(a_out_sia, 0)
        return pose_out_sia, a_out_sia

    def one_hot(self, src, num_classes=None, dtype=None):
        num_classes = src.max().item() + 1 if num_classes is None else num_classes
        out = torch.zeros(src.size(0), num_classes, dtype=dtype)
        out.scatter_(1, src.unsqueeze(-1).expand_as(out), 1)
        return out

    def spread_loss(self, x, target, epoch):
        target = self.one_hot(target[:,0].squeeze(), x.size(2)).cuda()
        m = torch.tensor(0.1, dtype=target.dtype,device=x.device)
        act_t1 = (x[0] * target).sum(dim=1)
        loss1 = ((F.relu(m - (act_t1.view(-1, 1) - x[0]))**2) * (1 - target))
        act_t2 = (x[1] * target).sum(dim=1)
        loss2 = ((F.relu(m - (act_t2.view(-1, 1) - x[1]))**2) * (1 - target))
        loss = loss1.sum(1).mean() + loss2.sum(1).mean()
        return loss

    def pose_diff_loss(self, pose):
        temp=torch.clamp(torch.abs((pose[0] * pose[1]).sum(dim=-1)),max=0.9999)           
        distance = 2*torch.acos(temp)/np.pi
        return distance.mean()
    
