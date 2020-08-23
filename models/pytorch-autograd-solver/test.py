import torch

import torch_autograd_solver as S
from torch.autograd import gradcheck,Variable
import torch.nn.functional as F
import random
def test_runtime():
    """test that there are no runtime errors"""
    import torch.nn as nn
    import torch.nn.functional as F
    x = torch.randn(30,10)
    w = nn.Parameter(torch.rand(30,10), requires_grad=True)
    xw = F.linear(x, w)
    a, b = S.symeig(xw)
    asum = a.sum()
    asum.backward()

def test_gradcheck():
    """test gradcheck"""
    input = torch.randn(5,5).double()
    input.requires_grad=True
    for upper in (True, False):
        assert gradcheck(S.BasicSymeig(upper=upper), (input,), eps=1e-6, atol=1e-4)

def test_symeig():
    # NOTE need pytorch 0.5.0 or 1.0
    a = torch.tensor([[ 1.96,  0.00,  0.00,  0.00,  0.00],
                      [-6.49,  3.80,  0.00,  0.00,  0.00],
                      [-0.47, -6.39,  4.17,  0.00,  0.00],
                      [-7.20,  1.50, -1.51,  5.70,  0.00],
                      [-0.65, -6.34,  2.67,  1.80, -7.10]]).t()
    a.requires_grad = True
    w, v = torch.symeig(a, eigenvectors=True)
    v.sum().backward()
    print(v.grad)


def test_batch_symeig_forward():
    xs = torch.randn(8, 4, 4).float().cuda()
    ws, vs = S.symeig(xs)
#    test=torch.symeig(xs[0])
    eigenValues, eigenVectors = torch.symeig(xs[0],eigenvectors=True)
    for i in range(xs.shape[0]):
        w, v = S.symeig(xs[i])
        torch.testing.assert_allclose(ws[i], w)
        torch.testing.assert_allclose(vs[i], v)

def test_batch_symeig_backward():
    input = torch.randn(8*64*32, 4, 4).float()
    input1 = input.clone()
    input.requires_grad = True
    w, v = S.symeig(input)
    (w.sum() + v.sum()).backward()
    # print(input.grad)
    for i in range(input1.size(0)):
        in1 = input1[i]
        in1.requires_grad = True
        wi, vi = S.symeig(in1)
        (wi.sum() + vi.sum()).backward()
        # print(in1.grad)
        torch.testing.assert_allclose(input.grad[i], in1.grad)

def test_batch_symeig_top():
#    input = torch.randn(8, 4, 4).float().cuda()
    num_q=800
    input_lrf = torch.randn(num_q, 8, 4).cuda()
    input_lrf=F.normalize(input_lrf, p=2, dim=-1)
    input_lrf=input_lrf.view(-1,4)
    test=torch.bmm(input_lrf.unsqueeze(2),input_lrf.unsqueeze(1))
    test=test.view(num_q,8,4,4)
    input =torch.sum(test,1)


    input3 = input.clone()
    input3.requires_grad = True
    
    input.requires_grad = True
    
    # batch wise back 
    w, v = S.symeig(input)    
    v_max=v[:,3].clone()
    bool_vmax=v_max[:,0]/torch.abs(v_max[:,0])
    bool_vmax=bool_vmax.contiguous().view(-1,1)      
    bool_vmax=bool_vmax.expand(v_max.size(0),4)  
    v_max=v_max*bool_vmax
    (v_max.mean()).backward()
    
    # pytorch version loop back 
    averaged_Q4=torch.rand(num_q,4)
    for i in range(input3.size(0)):
        eigenValues, eigenVectors = torch.eig(input3[i],eigenvectors=True)
        e_values, e_indices = torch.max(eigenValues, 0)            
        averaged_Q4[i]=eigenVectors[:,e_indices]
        if (averaged_Q4[i][0]<0):
            averaged_Q4[i]=-averaged_Q4[i]
    (averaged_Q4.mean()).backward()
    
    
    torch.testing.assert_allclose(input.grad, input3.grad)
    
def test_batch_symeig_top_init():
#    input = torch.randn(8, 4, 4).float().cuda()
    num_q=80
    
    
    input1 = torch.randn(num_q, 8, 4).cuda()
    input2=input1.clone()
    
    input1.requires_grad = True
    input_lrf=F.normalize(input1, p=2, dim=-1)
    input_lrf=input_lrf.view(-1,4)
    input_cov=torch.bmm(input_lrf.unsqueeze(2),input_lrf.unsqueeze(1))
    input_cov=input_cov.view(num_q,8,4,4)
    input_cov_sum =torch.sum(input_cov,1)
  
    w, v = S.symeig(input_cov_sum)      
    v_max=v[:,3].clone()
    
    bool_vmax=v_max[:,0]/torch.abs(v_max[:,0])
    bool_vmax=bool_vmax.contiguous().view(-1,1)      
    bool_vmax=bool_vmax.expand(v_max.size(0),4)  
    v_max=v_max*bool_vmax

    (v_max.mean()).backward()
    
#    input3.requires_grad = True
    averaged_Q4=torch.rand(num_q,4)
    input2.requires_grad = True
    input_lrf2=F.normalize(input2, p=2, dim=-1)
    input_lrf2=input_lrf2.view(-1,4)
    input_cov2=torch.bmm(input_lrf2.unsqueeze(2),input_lrf2.unsqueeze(1))
    input_cov2=input_cov2.view(num_q,8,4,4)
    input_cov_sum2 =torch.sum(input_cov2,1)
    for i in range(input_cov_sum2.size(0)):
        eigenValues, eigenVectors = torch.symeig(input_cov_sum2[i],eigenvectors=True)
        e_values, e_indices = torch.max(eigenValues, 0)            
        averaged_Q4[i]=eigenVectors[:,e_indices]
        if (averaged_Q4[i][0]<0):
            averaged_Q4[i]=-averaged_Q4[i]
    (averaged_Q4.mean()).backward()
    torch.testing.assert_allclose(input1.grad, input2.grad)    
    
#    
#    for i in range(input1.size(0)):
#        in1 = input1[i]
#        in1.requires_grad = True
#        wi, vi = S.symeig(in1)
#        e_v, e_i = torch.max(wi, 0)            
#        averaged_Q2=vi[:,e_i]
##        (averaged_Q2.sum()).backward()
#        (wi.sum() + vi.sum()).backward()
#    
#        in2 = input2[i]
#        in2.requires_grad = True
#        eigenValues, eigenVectors = torch.symeig(in2,eigenvectors=True)
#        e_values, e_indices = torch.max(eigenValues, 0)            
#        averaged_Q3=eigenVectors[:,e_indices]
##        averaged_Q3.sum().backward()
#        (eigenValues.sum() + eigenVectors.sum()).backward()
#
#        torch.testing.assert_allclose(input.grad[i], in2.grad)
#        torch.testing.assert_allclose(in1.grad, in2.grad)
#def test_generalized_symeig_forward():
#    a = torch.randn(3, 3).double()
#    a = a.t().mm(a)
#    b = torch.randn(3, 3).double()
#    b = b.t().mm(b)
#    w, v = S.symeig(a, b)
#    torch.testing.assert_allclose(a.mm(v), w * b.mm(v))
#
#def test_generalized_symeig_backward():
#    a = torch.randn(3, 3).double()
#    a = a.t().mm(a)
#    a.requires_grad=True
#    b = torch.randn(3, 3).double()
#    b = b.t().mm(b)
#    b.requires_grad=True
#    assert gradcheck(S.GeneralizedSymeig(), (a, b), eps=1e-6, atol=1e-4)


#test_symeig() # need pytorch 0.5.0 or later
#test_runtime()
#test_gradcheck()
#test_batch_symeig_forward()
        
def test_bug():
    test1=torch.tensor([[ 2.7378, -2.7378, -0.4963, -0.2952],
         [-2.7378,  2.7378,  0.4963,  0.2952],
         [-0.4963,  0.4963,  1.8648,  1.1091],
         [-0.2952,  0.2952,  1.1091,  0.6596]]).cuda()
#    test1=torch.randn( 4, 4).cuda()

#    test1_V = Variable(test1.data, requires_grad=True)
  
    test2=test1.unsqueeze(0)
    test2_V = Variable(test2.data, requires_grad=True)
    w, v = S.symeig(test2_V)
    loss=w.mean()+v.mean()
    loss.backward()
    print(test2_V.grad.max())
    
#    eigenValues, eigenVectors = torch.symeig(test1_V,eigenvectors=True)
#    loss=eigenValues.mean()+eigenVectors.mean()
#    loss.backward()
#    print(test1_V.grad.max())
    
    
    
manualSeed = 4041  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
#test_bug()s
test_batch_symeig_top_init()
#test_batch_symeig_top()
#test_batch_symeig_backward()
#test_generalized_symeig_forward()
#test_generalized_symeig_backward()
