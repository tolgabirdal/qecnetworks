#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:43:28 2019

"""

import torch
import torch.nn.functional as F
import torch_autograd_solver as S
from torch.nn import Module, Parameter, Linear

import numpy as np
import quat_ops
from open3d import *


USE_CUDA=True
class PoolingPointCapsuleLayer(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_iterations=3,num_neighbours=8,num_patches=8):
        super(PoolingPointCapsuleLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_iterations = num_iterations
        self.num_neighbours = num_neighbours
        self.num_patches = num_patches
        self.bottle_neck = in_channels*out_channels

        self.quater_gen1 =Linear(3 * self.in_channels, self.out_channels, bias=True)
        self.quater_gen2 =Linear(self.out_channels, self.out_channels * self.in_channels * 4, bias=True)
        
        self.alpha=Parameter(torch.FloatTensor(1))
        self.beta=Parameter(torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.alpha.data.fill_(1)
        self.beta.data.fill_(1)
#        self.c_gama.data.fill_(1)

    def averageQuaternions(self, _input_lrf, _input_a):
        num_q=_input_lrf.size(0)

        _input_lrf=_input_lrf.view(-1,4)
        cov_matrix=torch.bmm(_input_lrf.unsqueeze(2),_input_lrf.unsqueeze(1))
        cov_matrix=cov_matrix.view(num_q,self.num_neighbours,4,4)
        cov_matrix_ave =torch.mean(cov_matrix,1)

        _input_a=_input_a.transpose(-2,-1).contiguous().view(-1,self.num_neighbours)
        mask = torch.sign(torch.abs(_input_a))
        mask_4_matrix=torch.sum(mask,-1)
        mask_4_matrix=mask_4_matrix.nonzero().squeeze()
        cov_matrix_ave_none_zero=cov_matrix_ave[mask_4_matrix]

        noise=(1e-6)*torch.randn_like(cov_matrix_ave_none_zero).cuda()
        cov_matrix_ave_none_zero=cov_matrix_ave_none_zero+noise

        e_w, e_v = S.symeig(cov_matrix_ave_none_zero)
        v_max_=e_v[:,3].clone()
        v_max=torch.zeros((cov_matrix_ave.size(0),4),device=cov_matrix_ave.device,dtype=torch.float32)
        v_max[mask_4_matrix,:]=v_max_


        vmax_mask = torch.sign(v_max[:,0])
        vmax_mask=vmax_mask.contiguous().view(-1,1).expand(v_max.size(0),4)
        v_max=v_max*vmax_mask
        return v_max


    def weightedAverageQuaternions(self, pose, b_ij, input_a):
        weights=b_ij.float()
        mask = torch.sign(torch.abs(input_a))
        weights=weights*mask
        weights=F.normalize(weights, p=1, dim=-1)  # replace with softmax
        pose=pose.view(-1, self.num_neighbours*self.in_channels, 4)
        num_q=pose.size(0)

        mask_4_matrix=torch.sum(torch.sum(pose,-1),-1)
        mask_4_matrix=mask_4_matrix.nonzero().squeeze()

        pose=pose.view(-1,4)
        weights=weights.contiguous().view(-1,1,1)
        weights=weights.expand(pose.size(0),4,4).contiguous()

        cov_matrix=torch.bmm(pose.unsqueeze(2),pose.unsqueeze(1))

        weighted_cov=weights*cov_matrix
        weighted_cov=weighted_cov.view(num_q,self.num_neighbours*self.in_channels,4,4)
        cov_matrix_sum =torch.sum(weighted_cov,1)

        cov_matrix_sum_none_zero=cov_matrix_sum[mask_4_matrix]

        noise=(1e-6)*torch.randn_like(cov_matrix_sum_none_zero).cuda()
        cov_matrix_sum_none_zero=cov_matrix_sum_none_zero+noise

        e_w, e_v = S.symeig(cov_matrix_sum_none_zero)
        v_max_=e_v[:,3].clone()
        v_max=torch.zeros((cov_matrix_sum.size(0),4),device=cov_matrix_sum.device,dtype=torch.float32)
        v_max[mask_4_matrix,:]=v_max_
        vmax_mask = torch.sign(v_max[:,0])
        vmax_mask=vmax_mask.contiguous().view(-1,1).expand(v_max.size(0),4)
        v_max=v_max*vmax_mask

        return v_max


    def mean_lrf_per_patch(self,_input_lrf,input_a):
        batch_size=_input_lrf.size(0)
        _input_lrf=_input_lrf.transpose(-2,-3).contiguous().view(-1,self.num_neighbours,4)
        mean_lrf=self.averageQuaternions(_input_lrf, input_a)
        mean_lrf=mean_lrf.view(batch_size, self.num_patches, self.in_channels, 4)
        inverse_matrix=torch.FloatTensor([1.0,-1.0,-1.0,-1.0])
        if(USE_CUDA):
            inverse_matrix=inverse_matrix.cuda()
        inverse_matrix=inverse_matrix[None,None,None,:].expand_as(mean_lrf)
        return mean_lrf, mean_lrf*inverse_matrix


    def get_votes(self,lrf, mean_lrf_inv, input_a, points):
        batch_size=lrf.size(0)
        mean_lrf_inv=mean_lrf_inv.unsqueeze(-3).contiguous()
        mean_lrf_inv=mean_lrf_inv.expand(mean_lrf_inv.size(0), mean_lrf_inv.size(1),self.num_neighbours, mean_lrf_inv.size(3),mean_lrf_inv.size(4)).contiguous()

# transform the points
        points=points.unsqueeze(-2).expand(-1,-1,-1,self.in_channels,-1)
        rotated_constant_position=quat_ops.qrotv(mean_lrf_inv, points.contiguous())

# gnerate transformation matrix for dynamic routing
        t_ij1=self.quater_gen1(rotated_constant_position.view(-1, self.in_channels * 3).float())
        t_ij=self.quater_gen2(t_ij1).float()

        t_ij=t_ij.view(batch_size,self.num_patches,self.num_neighbours,4,self.in_channels,self.out_channels)
        t_ij=t_ij.transpose(-1,-3).contiguous()

        # normalize the transformation(quaternions) into unit quaternion
        t_ij=F.normalize(t_ij, p=2, dim=-1)

        # keep the first scalar of the quaternion positive.
        t_ij_mask = torch.sign(t_ij[:,:,:,:,:,0])
        t_ij_mask=t_ij_mask.contiguous().unsqueeze(-1).expand(-1,-1,-1,-1,-1,4)
        t_ij=t_ij*t_ij_mask

# transform the input LRF with output transformations from the linear network
        lrf = lrf[ :, :, :, None, :,:].expand_as(t_ij).contiguous()
#        v_ij=quat_ops.qmul(t_ij, lrf)# The left product  will break the Equ
        v_ij=quat_ops.qmul(lrf,t_ij)

        v_ij=v_ij.transpose(-3,-4).contiguous()
        v_ij= v_ij.view(batch_size,self.num_patches,self.out_channels,self.num_neighbours*self.in_channels, 4)

       # keep the first scalar of the quaternion positive.
        v_ij_mask = torch.sign(v_ij[:,:,:,:,0])
        v_ij_mask=v_ij_mask.contiguous().unsqueeze(-1).expand(-1,-1,-1,-1,4)
        v_ij=v_ij*v_ij_mask

        return v_ij

    def mean(self, vote, b_ij,input_a):
        batch_size=vote.size(0)
        weighted_mean_lrf= self.weightedAverageQuaternions(vote, b_ij,input_a)

        weighted_mean_lrf=weighted_mean_lrf.view(batch_size,self.num_patches,self.out_channels,4)
        return weighted_mean_lrf


    def distance(self, pose,vote):
        pose = pose[ :, :, :, None, :].expand_as(vote)
        temp=torch.clamp(torch.abs((pose * vote).sum(dim=-1)),max=0.9999)
        distance = 2*torch.acos(temp)/np.pi
        return distance

    def distance_naive(self, pose,vote):
        pose = pose[ :, :, :, None, :].expand_as(vote)
        distance = ((pose * vote).sum(dim=-1))**2
        return (-distance + 1)/2

    def forward(self, points, lrfs, activation):
        batch_size=lrfs.size(0)
        mean_lrf, mean_lrf_inv=self.mean_lrf_per_patch(lrfs, activation)
        vote=self.get_votes(lrfs, mean_lrf_inv,activation, points)

        vote_detached=vote.detach()
        activation = activation.view(batch_size,self.num_patches, self.num_neighbours * self.in_channels)
        activation = activation[ :, :, None, :].expand( -1, -1, self.out_channels, -1)
        b_ij = activation
        beta = self.beta.view(1, 1, -1)
        alpha = self.alpha.view( 1, 1, -1)
        if(USE_CUDA):
            b_ij=b_ij.cuda()

        for iteration in range(self.num_iterations):
            if iteration == self.num_iterations - 1:
                pose = self.mean(vote, b_ij, activation)
            else:
                pose = self.mean(vote_detached, b_ij, activation)
                b_ij =b_ij * activation * (1-self.distance(pose, vote_detached))

        neg_distance = b_ij *(1-self.distance(pose, vote))
        w_sum = torch.sign(torch.abs(b_ij)).sum(dim=-1)

        mask = torch.sign(torch.abs(w_sum))
        neg_distance = (neg_distance.sum(dim=-1) / (w_sum + (1 - mask))) * mask
        agreement = torch.sigmoid(alpha * neg_distance + (beta - 1)) * mask
        return pose.squeeze(-2), agreement
