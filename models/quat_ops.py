#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:31:58 2019

"""

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from numpy.random import RandomState
from scipy.stats import chi
#import sys
#from quat_ops import *
#import torch_autograd_solver as S
#import quat_ops
#from torch_batch_svd import batch_svd

# PyTorch-backed implementations

def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.contiguous().view(-1, 4, 1), q.contiguous().view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def qrotv(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qrotv3(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    # Compute outer product
    terms = torch.bmm(q.view(-1, 4, 1), q.view(-1, 1, 4))
    b2=terms[:,1,1]
    c2=terms[:,2,2]
    d2=terms[:,3,3]
    ab=terms[:,0,1]
    ac=terms[:,0,2]
    ad=terms[:,0,3]
    bc=terms[:,1,2]
    bd=terms[:,1,3]
    cd=terms[:,2,3]


    qvec_x=[1-2*c2-2*d2, 2*bc-2*ad, 2*ac+2*bd]
    qvec_y=[2*bc+2*ad, 1-2*b2-2*d2, 2*cd-2*ab]
    qvec_z=[2*bd-2*ac, 2*ab+2*cd, 1-2*b2-2*c2]
    qvec=torch.stack((torch.stack(qvec_x, dim=1), torch.stack(qvec_y, dim=1), torch.stack(qvec_z, dim=1)), dim=1)

    return torch.bmm(qvec,v.unsqueeze(-1)).view(original_shape)



def qrotq(q, p):
    """
    Rotate quaternion(s) p about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 4) for p,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
#    assert q.shape[-1] == 4
#    assert p.shape[-1] == 4
#    assert q.shape[:-1] == p.shape[:-1]

    original_shape = list(p.shape)
    q = q.view(-1, 4)
    p = p.view(-1, 4)
    pw=p[:,0]
    pv=p[:,1:4]

    qvec = q[:, 1:]
    uv = torch.cross(qvec, pv, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)

    pv=(pv + 2 * (q[:, :1] * uv + uuv))

#    return (pv + 2 * (q[:, :1] * uv + uuv)).view(original_shape)
    return torch.cat((pw.unsqueeze(-1), pv), dim=1).view(original_shape)

def qrotq3(q, p):
    """
    Rotate quaternion(s) p about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 4) for p,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert p.shape[-1] == 4
    assert q.shape[:-1] == p.shape[:-1]

    original_shape = list(p.shape)
    q = q.view(-1, 4)
    p = p.view(-1, 4)
    pw=p[:,0]
    pv=p[:,1:4]

    # Compute outer product
    terms = torch.bmm(q.view(-1, 4, 1), q.view(-1, 1, 4))
    b2=terms[:,1,1]
    c2=terms[:,2,2]
    d2=terms[:,3,3]
    ab=terms[:,0,1]
    ac=terms[:,0,2]
    ad=terms[:,0,3]
    bc=terms[:,1,2]
    bd=terms[:,1,3]
    cd=terms[:,2,3]


    qvec_x=[ 1-2*c2-2*d2, 2*bc-2*ad, 2*ac+2*bd]
    qvec_y=[ 2*bc+2*ad, 1-2*b2-2*d2, 2*cd-2*ab]
    qvec_z=[ 2*bd-2*ac, 2*ab+2*cd, 1-2*b2-2*c2]
    qvec=torch.stack((torch.stack(qvec_x, dim=1), torch.stack(qvec_y, dim=1), torch.stack(qvec_z, dim=1)), dim=1)

    pv=torch.bmm(qvec, pv.unsqueeze(-1)).squeeze()

    return torch.cat((pw.unsqueeze(-1), pv), dim=1).view(original_shape)

def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


def unitary_init(in_features, out_features, rng, kernel_size=None, criterion='he'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    s = np.sqrt(3.0) * s

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(-s,s,number_of_weights)
    v_i = np.random.uniform(-s,s,number_of_weights)
    v_j = np.random.uniform(-s,s,number_of_weights)
    v_k = np.random.uniform(-s,s,number_of_weights)



    # Unitary quaternion
    for i in range(0, number_of_weights):
        norm = np.sqrt(v_r[i]**2 + v_i[i]**2 + v_j[i]**2 + v_k[i]**2)+0.0001
        v_r[i]/= norm
        v_i[i]/= norm
        v_j[i]/= norm
        v_k[i]/= norm

    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    return (v_r, v_i, v_j, v_k)

def random_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    number_of_weights = np.prod(kernel_shape)
    v_r = np.random.uniform(0.0,1.0,number_of_weights)
    v_i = np.random.uniform(0.0,1.0,number_of_weights)
    v_j = np.random.uniform(0.0,1.0,number_of_weights)
    v_k = np.random.uniform(0.0,1.0,number_of_weights)

    v_r = v_r.reshape(kernel_shape)
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    weight_r = v_r * s
    weight_i = v_i * s
    weight_j = v_j * s
    weight_k = v_k * s
    return (weight_r, weight_i, weight_j, weight_k)


def quaternion_init(in_features, out_features, rng, kernel_size=None, criterion='glorot'):

    if kernel_size is not None:
        receptive_field = np.prod(kernel_size)
        fan_in          = in_features  * receptive_field
        fan_out         = out_features * receptive_field
    else:
        fan_in          = in_features
        fan_out         = out_features

    if criterion == 'glorot':
        s = 1. / np.sqrt(2*(fan_in + fan_out))
    elif criterion == 'he':
        s = 1. / np.sqrt(2*fan_in)
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState(np.random.randint(1,1234))


    # Generating randoms and purely imaginary quaternions :
    if kernel_size is None:
        kernel_shape = (in_features, out_features)
    else:
        if type(kernel_size) is int:
            kernel_shape = (out_features, in_features) + tuple((kernel_size,))
        else:
            kernel_shape = (out_features, in_features) + (*kernel_size,)

    modulus = chi.rvs(4,loc=0,scale=s,size=kernel_shape)

#    modulus= rng.uniform(size=kernel_shape)
    number_of_weights = np.prod(kernel_shape)


    v_i = np.random.normal(0,1.0,number_of_weights)
    v_j = np.random.normal(0,1.0,number_of_weights)
    v_k = np.random.normal(0,1.0,number_of_weights)

    # Purely imaginary quaternions unitary
    for i in range(0, number_of_weights):
    	norm = np.sqrt(v_i[i]**2 + v_j[i]**2 + v_k[i]**2 +0.0001)
    	v_i[i]/= norm
    	v_j[i]/= norm
    	v_k[i]/= norm
    v_i = v_i.reshape(kernel_shape)
    v_j = v_j.reshape(kernel_shape)
    v_k = v_k.reshape(kernel_shape)

    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)

    weight_r = modulus * np.cos(phase)
    weight_i = modulus * v_i*np.sin(phase)
    weight_j = modulus * v_j*np.sin(phase)
    weight_k = modulus * v_k*np.sin(phase)

    return (weight_r, weight_i, weight_j, weight_k)

def create_dropout_mask(dropout_p, size, rng, as_type, operation='linear'):
    if operation == 'linear':
        mask = rng.binomial(n=1, p=1-dropout_p, size=size)
        return Variable(torch.from_numpy(mask).type(as_type))
    else:
         raise Exception("create_dropout_mask accepts only 'linear'. Found operation = "
                        + str(operation))

def affect_init(q_weight, init_func, rng, init_criterion):
#    if r_weight.size() != i_weight.size() or r_weight.size() != j_weight.size() or \
#    r_weight.size() != k_weight.size() :
#         raise ValueError('The real and imaginary weights '
#                 'should have the same size . Found: r:'
#                 + str(r_weight.size()) +' i:'
#                 + str(i_weight.size()) +' j:'
#                 + str(j_weight.size()) +' k:'
#                 + str(k_weight.size()))
#
#    elif r_weight.dim() != 2:
#        raise Exception('affect_init accepts only matrices. Found dimension = '
#                        + str(r_weight.dim()))
    kernel_size = None
    r, i, j, k  = init_func(q_weight.size(0), q_weight.size(1), rng, kernel_size, init_criterion)
    r, i, j, k  = torch.from_numpy(r), torch.from_numpy(i), torch.from_numpy(j), torch.from_numpy(k)
    q_weight.data= torch.stack((r.type_as(q_weight.data),i.type_as(q_weight.data),j.type_as(q_weight.data),k.type_as(q_weight.data)),2)

#    r_weight.data = r.type_as(r_weight.data)
#    i_weight.data = i.type_as(i_weight.data)
#    j_weight.data = j.type_as(j_weight.data)
#    k_weight.data = k.type_as(k_weight.data)




if __name__ == '__main__':
    p=torch.rand(16,64,4)
    pool_grid=torch.FloatTensor([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
                            [-1.0, 1.0, -1.0],  [1.0, 1.0, -1.0],
                            [-1.0, -1.0, 1.0],  [1.0, -1.0, 1.0],
                            [-1.0, 1.0, 1.0],   [1.0, 1.0, 1.0]])
#
    p=p.unsqueeze(-2)
    p=p.expand(p.size(0), p.size(1),8, p.size(3)).contiguous()
    pool_grid=pool_grid.view(1,1,8,3)
    pool_grid=pool_grid.expand(p.size(0), p.size(1),8, 3).contiguous()

#    p=p.cuda()
#    pool_grid=pool_grid.cuda()
    test1=qrotv(p, pool_grid)


    test3=qrotv3(p, pool_grid)
    print(test1[0,0,])
    print(test3[0,0,])

    input_lrf=torch.rand(16,64,8,4)
    t_ij=torch.rand(16,64,8,32,4)
    input_lrf=input_lrf.unsqueeze(-2)
    input_lrf=input_lrf.expand(t_ij.size(0), t_ij.size(1), t_ij.size(2), 32, t_ij.size(4)).contiguous()
    test2=qrotq(t_ij, input_lrf)
    test4=qrotq3(t_ij, input_lrf)
