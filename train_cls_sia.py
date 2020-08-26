#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import numpy as np
import torch
import torch.optim as optim
from logger import Logger
sys.path.append('models')
sys.path.append('my_dataloader')
from qec_sia_net import QecSiaNet
import modelnet_with_lrf_sample_index_loader_sia
import time

def main():
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = modelnet_with_lrf_sample_index_loader_sia.ModelNetDataset(root=opt.data_path, npoints=opt.num_points,
                                                                              split='train', num_of_class=opt.num_of_class,
                                                                              class_choice=opt.class_choice,
                                                                              num_gen_samples=opt.num_gen_samples,data_aug=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    qec_net = QecSiaNet(opt.num_points, opt.inter_out_channels, opt.num_of_class, opt.num_iterations)

    if opt.model != '':
        qec_net.load_state_dict(torch.load(opt.model))

    print("Use", torch.cuda.device_count(), "GPUs!")
    
    qec_net = torch.nn.DataParallel(qec_net)
    qec_net=qec_net.cuda()

# create folder for log file
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_dir='./logs/'+timestr
    if not os.path.exists(log_dir):
        os.makedirs(log_dir);
    logger = Logger(log_dir)
    log_config = open(os.path.join(log_dir, 'config.txt'), 'w')
    log_config.write(str(vars(opt)))
    log_config.close()

    qec_net.train()
    
    for epoch in range(opt.n_epochs):

        if epoch <20:
            optimizer = optim.Adam(qec_net.parameters(), lr=opt.init_lr)
        elif epoch<40:
            optimizer = optim.Adam(qec_net.parameters(), lr=opt.init_lr/10)
        elif epoch<60:
            optimizer = optim.Adam(qec_net.parameters(), lr=opt.init_lr/100)
        else:
            optimizer = optim.Adam(qec_net.parameters(), lr=opt.init_lr/1000)

        train_cls_loss_sum = 0
        train_pose_loss_sum = 0
        batch_id=0
        
        
        
        for points_pool1,  lrfs_pool1, activation_pool1, points_pool2_index, target, point_set, _ in loader:
            if(int(points_pool1.size(0)) < opt.batch_size):
                continue
            points_pool1, lrfs_pool1, activation_pool1, points_pool2_index,point_set =points_pool1.cuda(), lrfs_pool1.cuda(), activation_pool1.cuda(), points_pool2_index.cuda(),point_set.cuda()
            optimizer.zero_grad()
            pose_out, a_out= qec_net(points_pool1, lrfs_pool1, activation_pool1, points_pool2_index)
            classify_loss =qec_net.module.spread_loss(a_out, target, epoch)
            pose_out_act=torch.zeros(2,opt.batch_size,4).cuda()

            # get the activate class from GT during training
            target_=target.squeeze()
            for i in range(opt.batch_size):
                pose_out_act[:,i]=pose_out[:,i,target_[i,0]]
            pose_loss = 0.1 * qec_net.module.pose_diff_loss(pose_out_act)
            (classify_loss + pose_loss).backward()
            optimizer.step()

            # for monitoring the training process
            train_cls_loss_sum += (classify_loss).item()
            train_pose_loss_sum += (pose_loss).item()
            info = {'classify_loss': classify_loss.item(), 'pose_loss': pose_loss.item()}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, (len(loader) * epoch) + batch_id + 1)
            if batch_id%5==0:
                print('bactch_no:[%d/%d/%d],  cls_loss and pose_loss are: %f,  %f ' %  (batch_id, len(loader),epoch, (classify_loss).item(), (pose_loss).item() ))
            batch_id+=1
        print('\033[94m Average train cls loss and pose loss of epoch %d : %f, %f \033[0m' % (epoch, (train_cls_loss_sum / len(loader)), (train_pose_loss_sum / len(loader))))    
        
        if epoch% 2 == 0:            
            dict_name=log_dir+'/tmp_'+str(epoch)+'.pth'
            torch.save(qec_net.module.state_dict(), dict_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--num_points', type=int, default=2048, help='input batch size')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--num_of_class', type=int, default=10, help='num_of_class')
    parser.add_argument('--data_path', type=str, default='/home/zhao/dataset/modelnet40_normal_resampled/', help='dataset path')
    parser.add_argument('--class_choice', type=str, default=None, help='chosse one cat')
    parser.add_argument('--num_gen_samples', type=int, default=30, help='num_gen_samples')
    parser.add_argument('--inter_out_channels', type=int, default=128, help='inter_out_channels')
    parser.add_argument('--num_iterations', type=int, default=3, help='num_iterations')
    parser.add_argument('--init_lr', type=float, default= 0.001, help='init_lr')

    opt = parser.parse_args()
    print(opt)

    main()
