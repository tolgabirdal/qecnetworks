#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
import sys
sys.path.append('models')
sys.path.append('my_dataloader')
import modelnet_with_lrf_sample_index_loader_sia
from qec_sia_net import QecSiaNet
import quat_ops
import torch_autograd_solver as S


def ave_pose(pose):
    batch_size=pose.size(0)
    ave_out=torch.zeros(batch_size,4)
    for b_id in range (batch_size):
        mask_4_pose=torch.sum(pose[b_id ],-1).nonzero().squeeze()
        pose_none_zero=pose[b_id, mask_4_pose]
        cov_matrix=torch.bmm(pose_none_zero.unsqueeze(2),pose_none_zero.unsqueeze(1))
        cov_matrix_ave =torch.mean(cov_matrix,0)
        e_w, e_v = S.symeig(cov_matrix_ave)
        v_max=e_v[:,3].clone()
        vmax_mask = torch.sign(v_max[0])
        vmax_mask=vmax_mask.contiguous().view(-1,1).expand(1,4)
        ave_out[b_id]=v_max * vmax_mask
    return ave_out


def q_distance(pose1,pose2):
    pose1_mask = torch.sign(pose1[0])
    pose1_mask=pose1_mask.contiguous().expand(4)
    pose1=pose1*pose1_mask
    pose2_mask = torch.sign(pose2[0])
    pose2_mask=pose2_mask.contiguous().expand(4)
    pose2=pose2*pose2_mask
    temp=torch.clamp(torch.abs((pose1 * pose2).sum(dim=-1)),max=1.0)
    distance = 2*torch.acos(temp)/np.pi
    return distance


def average_points_relative_error(points,relative_pose):
    points_x_none_zero_id=points[:,0].nonzero().squeeze()
    points=points[points_x_none_zero_id,:]
    points_y_none_zero_id=points[:,1].nonzero().squeeze()
    points=points[points_y_none_zero_id,:]
    points_z_none_zero_id=points[:,2].nonzero().squeeze()
    points=points[points_z_none_zero_id,:]

    relative_pose_=relative_pose.unsqueeze(0).expand(points.size(0),4)
    relative_points=quat_ops.qrotv(relative_pose_, points)

    relative_error=abs(torch.norm((relative_points-points))/(torch.norm(points)+torch.norm(relative_points)+ 1e-9))

    return torch.mean(relative_error)

def main():
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    qec_net = QecSiaNet(opt.num_points, opt.inter_out_channels, opt.num_of_class, opt.num_iterations)

    if opt.model != '':
        qec_net.load_state_dict(torch.load(opt.model))

    if USE_CUDA:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        qec_net.to(device)

    qec_net.eval()
    
    input_error=0
    acc_count=0
    act_error=0
    ave_relative_error_all=0
    sym_act_error=0
    pose_out_act=torch.zeros(2,opt.batch_size,4).cuda()

    dataset =modelnet_with_lrf_sample_index_loader_sia.ModelNetDataset(root=opt.data_path, npoints=opt.num_points,
                                                                       split='test',num_of_class=5, class_choice=opt.class_choice,
                                                                       num_gen_samples=opt.num_gen_samples, data_aug=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=1)
    our_pose_error_all=torch.zeros(908,7).float()
    batch_id=0
    for points_pool1,  lrfs_pool1, activation_pool1, points_pool2_index, target, point_set, rot_quat_gt in loader:
        points_pool1, lrfs_pool1, activation_pool1, points_pool2_index,point_set =points_pool1.cuda(), lrfs_pool1.cuda(), activation_pool1.cuda(), points_pool2_index.cuda(),point_set.cuda()
        pose_out, a_out= qec_net(points_pool1, lrfs_pool1,  activation_pool1, points_pool2_index)

        pose_out_act=torch.zeros(2,opt.batch_size,4).cuda()
        target_=target.squeeze()

        inverse_matrix=torch.FloatTensor([1.0,-1.0,-1.0,-1.0]).cuda()
        inverse_matrix_=inverse_matrix.unsqueeze(0).expand(points_pool1.size(0),4)
        inverse_rot_gt=rot_quat_gt.cuda() * inverse_matrix_

        for i in range(points_pool1.size(0)):
            shape_id=opt.batch_size * batch_id +i

            _,a_pred = a_out[:,i].cpu().max(1)
            our_pose_error_all[shape_id,0]=float(target_[i,0].item())
            our_pose_error_all[shape_id,1]=float(a_pred[0].item())
            our_pose_error_all[shape_id,2]=float(a_pred[1].item())

            if(target_[i,0]==a_pred[0].data and target_[i,0]==a_pred[1].data):
                acc_count+=1
                lrfs_pool1_c_4_ave=lrfs_pool1[i].view(2, -1,4)
                ave_input=ave_pose(lrfs_pool1_c_4_ave).cpu()
                ave_input_error=q_distance(ave_input[0], ave_input[1])

                our_pose_error_all[shape_id,3]=ave_input_error.item()
                input_error+=ave_input_error.item()

                pose_out_act[0,i]=pose_out[0,i,a_pred[0].data]
                pose_out_act[1,i]=pose_out[1,i,a_pred[1].data]
                pose_out_act[1,i]= quat_ops.qmul(inverse_rot_gt[i],pose_out_act[1,i])

                active_pose_error=q_distance(pose_out_act[0,i], pose_out_act[1,i])
                our_pose_error_all[shape_id,4]=active_pose_error.item()
                act_error+=active_pose_error.item()

                # for symetric
                pose_out_act0_1=quat_ops.qmul( torch.tensor([0,0,1,0]).float().cuda(),pose_out_act[0,i])
                active_pose_error2=q_distance(pose_out_act0_1, pose_out_act[1,i])
                if(active_pose_error>active_pose_error2):
                    sym_act_error+=active_pose_error2.item()
                    our_pose_error_all[shape_id,5]=active_pose_error2.item()
                else:
                    sym_act_error+=active_pose_error.item()
                    our_pose_error_all[shape_id,5]=active_pose_error.item()

                # relative error
                inv_rotate_q=pose_out_act[0,i]*inverse_matrix
                relative_active_pose=quat_ops.qmul(inv_rotate_q,pose_out_act[1,i])
                ave_relative_error=average_points_relative_error(point_set[0], relative_active_pose)
                ave_relative_error_all+=ave_relative_error.item()
                our_pose_error_all[shape_id,6]=ave_relative_error.item()

        batch_id+=1

    input_error=input_error/acc_count
    sym_act_error=sym_act_error/acc_count
    act_error=act_error/acc_count
    ave_relative_error_ave=ave_relative_error_all/acc_count

    print('ave_input_error ', input_error)
    print('active_pose_error ', act_error)
    print('sym_act_pose_error_ave ', sym_act_error)
    print('acc cound: ', acc_count)
    print('ave_relative_error_ave ', ave_relative_error_ave)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs to train for')
    parser.add_argument('--num_points', type=int, default=2048, help='input point size')
    parser.add_argument('--model', type=str, default='checkpoints/tmp_2.pth', help='model path')
    parser.add_argument('--num_of_class', type=int, default=10, help='num_of_class')
    parser.add_argument('--data_path', type=str, default='/home/zhao/dataset/modelnet40_normal_resampled/', help='dataset path')
    parser.add_argument('--num_gen_samples', type=int, default=5, help='num_gen_samples')
    parser.add_argument('--class_choice', type=str, default='chair', help='chosse one cat')
    parser.add_argument('--inter_out_channels', type=int, default=128, help='inter_out_channels')
    parser.add_argument('--num_iterations', type=int, default=1, help='num_iterations')
    opt = parser.parse_args()
    print(opt)
    main()
