# coding: utf-8
import numpy
import torch
import torch_cusolver

import torch.nn.functional as F

def test_batch_eigh():
#    A = torch.rand(8, 4, 4).cuda()
#    A = A.transpose(1, 2).matmul(A)
    
    num_q=8
    input_lrf = torch.randn(num_q, 8, 4).cuda()
    input_lrf=F.normalize(input_lrf, p=2, dim=-1)
    input_lrf=input_lrf.view(-1,4)
    test=torch.bmm(input_lrf.unsqueeze(2),input_lrf.unsqueeze(1))
    test=test.view(num_q,8,4,4)
    A =torch.sum(test,1)
    A.requires_grad=True
    w, V = torch_cusolver.cusolver_batch_eigh(A,
                                              False,
                                              True,
                                              1e-7,
                                              100,
                                              True)
    (w.sum()+V.sum()).backward()
#    test=torch.symeig(A[0])
    for i in range(A.shape[0]):
        a = A[i]
        e = V[i].t().matmul(w[i].diag()).matmul(V[i])
        torch.testing.assert_allclose(a, e)


#def test_generalized_eigh():
#    # A = torch.rand(3, 3).cuda()
#    # A = A.transpose(0, 1).matmul(A)
#    # B = torch.rand(3, 3).cuda()
#    # B = B.transpose(0, 1).matmul(B)
#    # example from https://docs.nvidia.com/cuda/cusolver/index.html#sygvd-example1
#    A = torch.cuda.FloatTensor(
#        [[3.5, 0.5, 0.0],
#         [0.5, 3.5, 0.0],
#         [0.0, 0.0, 2.0]])
#    B = torch.cuda.FloatTensor(
#        [[10, 2, 3],
#         [2, 10, 5],
#         [3, 5, 10]])
#    w_expect = torch.cuda.FloatTensor([0.158660256604, 0.370751508101882, 0.6])
#    for upper in [True, False]:
#        for jacob in [True, False]:
#            w, V, L = torch_cusolver.cusolver_generalized_eigh(A, False, B, False, upper, jacob, 1e-7, 100)
#            torch.testing.assert_allclose(w, w_expect)
#            torch.testing.assert_allclose(V.mm(B).mm(V.t()), torch.eye(A.shape[0], device=A.device))
#            for i in range(3):
#                torch.testing.assert_allclose(A.matmul(V[i]), B.matmul(V[i]) * w[i])


#def test_batch_svd():
#    # example from https://docs.nvidia.com/cuda/cusolver/index.html#batchgesvdj-example1
#    A = torch.cuda.FloatTensor(
#        [[[ 1, -1],
#          [-1,  2],
#          [ 0,  0]],
#         [[3, 4],
#          [4, 7],
#          [0, 0]]]) # .transpose(1, 2).contiguous()
#    s_expect = torch.cuda.FloatTensor(
#        [[2.6180, 0.382],
#         [9.4721, 0.5279]])
#    U, s, V = torch_cusolver.cusolver_batch_svd(A, False, 0.0, 100)
#
#    # FIXME not matched
#    print(s_expect)
#    print(s)
#
#    # # s (2, 2) -> (2, 3)
#    for i in range(A.shape[0]):
#        spad = torch.zeros(3, 2, device=A.device)
#        spad.diagonal()[:2] = s[i]
#        print(i)
#        # FIXME not matched
#        print(A[i])
#        print(U[i].t().mm(spad).mm(V[i]))
#

#def test_batch_matinv():
#    a = torch.randn(2, 3, 3).cuda()
#    ai = torch_cusolver.cusolver_batch_matinv(a)
#    for i in range(a.shape[0]):
#        torch.testing.assert_allclose(a[i].mm(ai[i]), torch.eye(a.shape[1], device=a.device))
#

#def test_complex_gemm():
#    torch.manual_seed(0)
#    for d in ["cpu", "cuda"]:
#        dev = torch.device(d)
#        a = torch.randn(4, 3, 2).to(dev)
#        b = torch.randn(3, 2, 2).to(dev)
#        c = torch_cusolver.cublas_cgemm(a, b)
#        for i in range(c.shape[0]):
#            for j in range(c.shape[1]):
#                ai = a[i, :].t()
#                bj = b[:, j].t()
#                # ar * br - ai * bi
#                cr = sum(ai[0] * bj[0] - ai[1] * bj[1])
#                # ai * br + ar * bi
#                ci = sum(ai[1] * bj[0] + ai[0] * bj[1])
#                torch.testing.assert_allclose(c[i, j, 0], cr)
#                torch.testing.assert_allclose(c[i, j, 1], ci)

test_batch_eigh()
#test_generalized_eigh()
#test_batch_matinv()
#test_batch_svd()
#test_complex_gemm()
#