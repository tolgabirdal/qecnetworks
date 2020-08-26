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


# In[Two pooling layers ]
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


    # def forward(self, points4pool1, lrfs4pool1,  activation4pool1, pointspool1index):
    #     batch_size=points4pool1.size(0)
    #     lrfs_4_pool1=lrfs4pool1.unsqueeze(-2)
    #     activation_4_pool1=activation4pool1.unsqueeze(-1)

    # # first pooling
    #     pose_pool1, activation_pool1 = self.pool1(points4pool1, lrfs4pool1, activation4pool1)
    # # generate centers and neighbours for the 2rd pooling.
    #     points4pool2=points4pool1[:,:,0,:].unsqueeze(1)# replace this one with the center point
    #     pose4pool2=pose_pool1.unsqueeze(1)
    #     activation4pool2=activation_pool1.unsqueeze(1)
    # # second pooling
    #     pose_pool2, activation_pool2= self.pool2(points4pool2, pose4pool2, F.relu(activation4pool2))
    #     return pose_pool2.squeeze(), activation_pool2.squeeze()

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


# # import os
# # import torch
# # import torch.nn as nn
# # import torch.nn.parallel
# # import torch.utils.data
# # from torch.autograd import Variable
# # import torch.nn.functional as F
# # from collections import OrderedDict
# # import sys
# # from pooling_point_capsule_layer import PoolingPointCapsuleLayer
# # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# # from torch.autograd.gradcheck import gradgradcheck, gradcheck
# # import healpy as hp
# # USE_CUDA = True
# # import numpy as np

# # import quat_ops
# # import torch_autograd_solver as S
# # #import time
# # import matplotlib.pyplot as plt

# # import torch_nndistance as NND
# import torch
# import torch.nn as nn
# import numpy as np
# import quat_ops
# import torch.nn.functional as F
# import torch_autograd_solver as S
# from qec_module import QecModule




# class PoseNet2(torch.nn.Module):
#     def __init__(self,num_points,inter_out_channels, num_of_class, num_iterations):
#         super(PoseNet2, self).__init__()
#         self.num_points=num_points
#         self.inter_out_channels=inter_out_channels
# #        self.final_out_channels=final_out_channels
#         self.num_iterations=num_iterations
#         self.num_of_class=num_of_class
        
# #        self.num_pool1_patches=1024
#         self.num_pool2_patches=256
#         self.num_pool3_patches=1
#         self.num_pool4_patches=1
    
#         self.num_neighbours2=9
# #        self.num_neighbours2=81
#         self.num_neighbours3=256
#         self.num_neighbours4=1
        
# #        self.pool1=PoolingPointCapsuleLayer(in_channels=1, out_channels=inter_out_channels, num_iterations=num_iterations, num_neighbours=self.num_neighbours1, num_patches=self.num_pool1_patches)
#         self.pool2=PoolingPointCapsuleLayer(in_channels=1, out_channels=inter_out_channels, num_iterations=num_iterations, num_neighbours=self.num_neighbours2, num_patches=self.num_pool2_patches)
#         self.pool3=PoolingPointCapsuleLayer(in_channels=inter_out_channels, out_channels=self.num_of_class, num_iterations=num_iterations, num_neighbours=self.num_neighbours3, num_patches=self.num_pool3_patches)
        
#         # final layer is just a dynamic routing. 1 point with 32 capsulpoints_4_pool2es to 1 point with 10(number of classes), the point is set to [0,0,0].So it is basically last layers is only based on bias.
# #        self.pool4=PoolingPointCapsuleLayer(in_channels=final_out_channels, out_channels=num_of_class, num_iterations=num_iterations, num_neighbours=self.num_neighbours4, num_patches=self.num_pool4_patches)
  
#     def forward(self, points_4_pool2, lrfs_4_pool2,  activation_4_pool2, points_pool2_index):      
#         batch_size=points_4_pool2.size(0)
#         lrfs_4_pool2=lrfs_4_pool2.unsqueeze(-2)
#         activation_4_pool2=activation_4_pool2.unsqueeze(-1)
    
#     # second pooling         
#         pose_pool2, activation_pool2 = self.pool2(points_4_pool2, lrfs_4_pool2, activation_4_pool2)
         
#     # generate centers and neighbours for the 3rd pooling.         
#         points_4_pool3=points_4_pool2[:,:,0,:].unsqueeze(1)# replace this one with the center point                
#         # use the local position rather than global
# #        w_sum = torch.sign(torch.abs(points_4_pool2.sum(-1))).sum(dim=-1).unsqueeze(-1)
# #        mask = torch.sign(torch.abs(w_sum))
# #        points_center=(points.sum(dim=-2)/ (w_sum + (1 - mask))) * mask
# #        points=points-points_center.unsqueeze(-2)
        
#         pose_4_pool3=pose_pool2.unsqueeze(1)
#         activation_4_pool3=activation_pool2.unsqueeze(1)
#     # third pooling 
#         pose_pool3, activation_pool3= self.pool3(points_4_pool3, pose_4_pool3, F.relu(activation_4_pool3))
    
#         return pose_pool3.squeeze(), activation_pool3.squeeze()
    

 
# class PointCapsNet(nn.Module):
#     def __init__(self, num_points, inter_out_channels, num_of_class, num_iterations ):
#         super(PointCapsNet, self).__init__()
#         self.num_points=num_points    
#         self.inter_out_channels=inter_out_channels
#         self.num_iterations =num_iterations
#         self.num_of_class=num_of_class
#         self.pose_net=PoseNet2(num_points,inter_out_channels, num_of_class, num_iterations)

#     def forward(self, points_pool1, lrfs_pool1,  activation_pool1, points_pool2_index):
#         pose_out_2=[]
#         a_out_2=[]
#         for i in range(2):
#             pose_out, a_out=self.pose_net(points_pool1[:,i], lrfs_pool1[:,i],  activation_pool1[:,i], points_pool2_index[:,i])     
#             pose_out_2.append(pose_out)
#             a_out_2.append(a_out)
        
#         pose_out_2=torch.stack(pose_out_2,0)
#         a_out_2=torch.stack(a_out_2, 0)
#         return pose_out_2, a_out_2
    
#     def one_hot(self, src, num_classes=None, dtype=None):
#         num_classes = src.max().item() + 1 if num_classes is None else num_classes
#         out = torch.zeros(src.size(0), num_classes, dtype=dtype)
#         out.scatter_(1, src.unsqueeze(-1).expand_as(out), 1)
#         return out
    
#     def spread_loss(self, x, target, epoch):
        
#         target = self.one_hot(target[:,0].squeeze(), x.size(2)).cuda()
# #        m = min(0.1 + 0.05 * (epoch+1), 0.9)
#         m= 0.1
#         m = torch.tensor(m, dtype=target.dtype,device=x.device)
        
#         act_t1 = (x[0] * target).sum(dim=1)
#         loss1 = ((F.relu(m - (act_t1.view(-1, 1) - x[0]))**2) * (1 - target))
        
#         act_t2 = (x[1] * target).sum(dim=1)
#         loss2 = ((F.relu(m - (act_t2.view(-1, 1) - x[1]))**2) * (1 - target))
        
# #        loss=F.normalize(loss,p=2, dim=-1)
#         loss = loss1.sum(1).mean() + loss2.sum(1).mean()
#         return loss
    
    
#     def pose_diff_loss(self, pose):
#         temp=torch.clamp(torch.abs((pose[0] * pose[1]).sum(dim=-1)),max=0.9999)           
#         distance = 2*torch.acos(temp)/np.pi
#         return distance.mean()
    
