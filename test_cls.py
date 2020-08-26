#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
import random
import sys
sys.path.append('models')
sys.path.append('my_dataloader')
from qec_net import QecNet
import modelnet_with_lrf_sample_index_loader
import matplotlib.pyplot as plt
def main():

    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opt.manualSeed = random.randint(1, 10000)  # fix seed
#    opt.manualSeed =9999 # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    dataset = modelnet_with_lrf_sample_index_loader.ModelNetDataset(root=opt.data_path, npoints=opt.num_points,
                                                                          split='test', num_of_class=opt.num_of_class,
                                                                          class_choice=opt.class_choice,
                                                                          num_gen_samples=opt.num_gen_samples, data_aug=opt.data_aug,
                                                                          point_shift=False,rand_seed=opt.manualSeed)

    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    qec_net = QecNet(opt.num_points, opt.inter_out_channels, opt.num_of_class, opt.num_iterations)

    if opt.model != '':
        qec_net.load_state_dict(torch.load(opt.model))

    if USE_CUDA:
        qec_net.to(device)
        
    qec_net.eval()
    eq_sum=0
    batch_id=0
    cat_results=torch.zeros(2,opt.num_of_class)

    cat_results2=torch.zeros(opt.num_of_class,opt.num_of_class)
    for points_pool1,  lrfs_pool1, activation_pool1, points_pool2_index, _,target in loader:
        if(points_pool1.dim()<4):
            points_pool1,  lrfs_pool1, activation_pool1, points_pool2_index, target=points_pool1.unsqueeze(0),  lrfs_pool1.unsqueeze(0), activation_pool1.unsqueeze(0), points_pool2_index.unsqueeze(0), target.unsqueeze(0)

        cur_bs=target.size(0)
        target=target.squeeze(-1).squeeze(-1)
        points_pool1,  lrfs_pool1,  activation_pool1,points_pool2_index =points_pool1.cuda(), lrfs_pool1.cuda(), activation_pool1.cuda(), points_pool2_index.cuda()
        pose_out, a_out= qec_net(points_pool1, lrfs_pool1,  activation_pool1, points_pool2_index)
        if(pose_out.dim()<3):
            pose_out, a_out=pose_out.unsqueeze(0), a_out.unsqueeze(0)
        a_pred = a_out.max(1)[1]
        for i in range(cur_bs):
            cat_results[0,target[i]]+=1
            if(target[i]==a_pred[i].data.cpu()):
                cat_results[1,target[i]]+=1
#                else:
#                    print('pred of file with index of %d with label %s is %s : ' % (batch_id*(opt.batch_size)+i , cat_no[target[i]], cat_no[a_pred[i].data.cpu()] ) )
            cat_results2[target[i],a_pred[i]]+=1
        batch_id+=1
    for i in range(opt.num_of_class)   :
        cat_results2[i]=cat_results2[i]/cat_results[0,i]
    plt.figure()
    plt.imshow(cat_results2)
    plt.colorbar()
    plt.show()
    print('result is : ' , cat_results[1].sum()/cat_results[0].sum() )
    cat_results[1]=cat_results[1]/cat_results[0]
    print(cat_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
    parser.add_argument('--model', type=str, default='checkpoints/tmp_4.pth', help='model path')
    parser.add_argument('--num_of_class', type=int, default=10, help='num_of_class')
    parser.add_argument('--class_choice', type=str, default=None, help='chosse one cat')
    parser.add_argument('--data_path', type=str, default='/home/zhao/dataset/modelnet40_normal_resampled/', help='dataset path')
    parser.add_argument('--num_gen_samples', type=int, default=5, help='num_gen_samples')
    parser.add_argument('--data_aug',  type=bool, default=False, help='If rotate the shape')
    parser.add_argument('--inter_out_channels', type=int, default=128, help='inter_out_channels')
    parser.add_argument('--num_iterations', type=int, default=3, help='num_iterations')
    opt = parser.parse_args()
    print(opt)

    main()
