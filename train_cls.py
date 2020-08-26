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
from qec_net import QecNet
import modelnet_with_lrf_sample_index_loader
import time

def main():        
    dataset = modelnet_with_lrf_sample_index_loader.ModelNetDataset(root=opt.data_path, npoints=opt.num_points,
                                                                          split='train', num_of_class=opt.num_of_class,
                                                                          class_choice=opt.class_choice,
                                                                          num_gen_samples=opt.num_gen_samples,data_aug=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    qec_net = QecNet(opt.num_points, opt.inter_out_channels, opt.num_of_class, opt.num_iterations)
    
    if opt.model != '':
        qec_net.load_state_dict(torch.load(opt.model))
    
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
#    with autograd.detect_anomaly(): # this is used to check the grad error
    for epoch in range(opt.n_epochs):

        if epoch <10:
            optimizer = optim.Adam(qec_net.parameters(), lr=opt.init_lr)
        elif epoch<30:
            optimizer = optim.Adam(qec_net.parameters(), lr=opt.init_lr/10)
        elif epoch<40:
            optimizer = optim.Adam(qec_net.parameters(), lr=opt.init_lr/100)
        else:
            optimizer = optim.Adam(qec_net.parameters(), lr=opt.init_lr/1000)

        train_loss_sum = 0
        batch_id=0
        
        for points_pool1,  lrfs_pool1, activation_pool1, points_pool2_index,_, target in loader:
            if(int(points_pool1.size(0)) < opt.batch_size):
                continue

            points_pool1, lrfs_pool1, activation_pool1, points_pool2_index =points_pool1.cuda(), lrfs_pool1.cuda(), activation_pool1.cuda(), points_pool2_index.cuda()
            optimizer.zero_grad()
            pose_out, a_out= qec_net(points_pool1, lrfs_pool1,  activation_pool1, points_pool2_index)

            classify_loss =qec_net.module.spread_loss(a_out, target,epoch)
            classify_loss.backward()

            optimizer.step()
            
            # for monitoring the training process
            train_loss_sum += (classify_loss).item()
            info = {'classify_loss': classify_loss.item()}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, (len(loader) * epoch) + batch_id + 1)

            if batch_id % 5 == 0:
                print('bactch_no:[%d/%d/%d],  cls_loss: %f ' %  (batch_id, len(loader),epoch, classify_loss.item() ))
                if batch_id % 100 == 0:
                    for tag, value in qec_net.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, value.data.cpu().numpy(), (len(loader) * epoch) + batch_id + 1)
                        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), (len(loader) * epoch) + batch_id + 1)
                    a_pred = a_out.max(1)[1]
                    eq1 = (a_pred.data.cpu()).eq(target.squeeze())
                    print('\x1b[6;30;42m bactch_no:[%d/%d/%d], cls_acc: %f \x1b[0m' %  (batch_id, len(loader), epoch, float(eq1.sum().item()/(opt.batch_size)) ))
            batch_id+=1
        print('\033[94m Average train loss of epoch %d : %f \033[0m' % (epoch, (train_loss_sum / len(loader))))
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
    parser.add_argument('--num_gen_samples', type=int, default=30, help='num_gen_samples')# generate 30 different downsamples from each input point cloud.
    parser.add_argument('--inter_out_channels', type=int, default=128, help='inter_out_channels') # number of hiden units in T-net.
    parser.add_argument('--num_iterations', type=int, default=3, help='num_iterations') # number of iterations of DR.
    parser.add_argument('--init_lr', type=float, default= 0.001, help='init_lr')
    opt = parser.parse_args()
    print(opt)
    main()
