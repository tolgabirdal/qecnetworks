# -*- coding: utf-8 -*-

import torch
#import torch
import torch_autograd_solver as S
import time

def test():
    seconds = time.time()
    for i in range(10):
        a = torch.rand(65535, 4, 4).cuda()
    #    b = a.clone()
    #    c = a.clone()
        a.requires_grad = True
    #    b.requires_grad = True
    #    c.requires_grad = True

    

        U, V = S.symeig(a)
        loss = U.mean() + V.mean()
        loss.backward()
    
    
    seconds2 = time.time()

    print("time of eig ", seconds2-seconds)	
    
if __name__ == '__main__':
    test()

    print('Finished')
