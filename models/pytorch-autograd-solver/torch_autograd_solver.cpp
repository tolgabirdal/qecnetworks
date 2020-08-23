#include <torch/torch.h>

#include <iostream>

/*
  Suppose A is square matrix, solve A V = V W, where V is unitary and W is diagonal
  Its differentiation gives dL/dA = V (dL/dW + F * (V^T dL/dV)) V^T, where F_{i,j, i \neq j} = 1/(W_i - W_j)

  dA V + A dV = dV W + W dV
  dA V + V W dC = V dC W + V dW   (define dC = V^-1 dV <-> dV = V dC)

 */
// forked from pytorch 0.5
// https://github.com/sethah/pytorch/blob/81b61db9219ffeb8fc0c8ab3abe0f0b5a7edf4f4/tools/autograd/templates/Functions.cpp#L1514
// http://eprints.maths.ox.ac.uk/1079/1/NA-08-01.pdf
at::Tensor symeig_backward(
    // backward variables
    const at::Tensor& grad_loss_wrt_eigenvalues, // [m]
    const at::Tensor& grad_loss_wrt_eigenvectors, // [m, m]
    // forward variables
    const at::Tensor& a, // [m, m]
    const at::Tensor& eigenvalues, // [m]
    const at::Tensor& eigenvectors, // [m, m]
    // config
    bool upper)
{
    auto m = a.size(0);
    AT_CHECK(a.dim() == 2, "not square input matrix");
    AT_CHECK(a.size(1) == m, "not square input matrix");

    AT_CHECK(eigenvalues.dim() == 1, "invalid eigenvalues shape");
    AT_CHECK(eigenvalues.size(0) == m, "invalid eigenvalues shape");
    AT_CHECK(eigenvectors.dim() == 2, "invalid eigenvectors shape");
    AT_CHECK(eigenvectors.size(0) == m, "invalid eigenvectors shape");
    AT_CHECK(eigenvectors.size(1) == m, "invalid eigenvectors shape");

    AT_CHECK(grad_loss_wrt_eigenvalues.dim() == 1, "invalid grad_loss_wrt_eigenvalues shape");
    AT_CHECK(grad_loss_wrt_eigenvalues.size(0) == m, "invalid grad_loss_wrt_eigenvalues shape");
    AT_CHECK(grad_loss_wrt_eigenvectors.dim() == 2, "invalid grad_loss_wrt_eigenvectors shape");
    AT_CHECK(grad_loss_wrt_eigenvectors.size(0) == m, "invalid grad_loss_wrt_eigenvectors shape");
    AT_CHECK(grad_loss_wrt_eigenvectors.size(1) == m, "invalid grad_loss_wrt_eigenvectors shape");

    at::Tensor ga; // [m, m]
    auto vt = eigenvectors.t();
    if (grad_loss_wrt_eigenvectors.defined())
    {
        at::Tensor F = eigenvalues.unsqueeze(0).expand_as(a).clone(); // [m, m]
        F.sub_(at::unsqueeze(eigenvalues, 1));
        F.diagonal().fill_(INFINITY);
        F.reciprocal_();
        F.mul_(vt.mm(grad_loss_wrt_eigenvectors));
        ga = eigenvectors.mm(F.mm(vt));
    }
    if (grad_loss_wrt_eigenvalues.defined())
    {
        auto ga_gw = (eigenvectors * grad_loss_wrt_eigenvalues).mm(vt);
        if (ga.defined()) {
            ga.add_(ga_gw);
        } else {
            ga = ga_gw;
        }
    }
    if (upper)
    {
        auto gau = at::triu(ga.t(), 1);
        ga.triu_().add_(gau);
    }
    else
    {
        auto gal = at::tril(ga.t(), -1);
        ga.tril_().add_(gal);
    }
    return ga;
}

std::tuple<at::Tensor, at::Tensor> batch_symeig_forward(const at::Tensor& input, bool upper)
{
    auto batch_size = input.size(0);
    auto n = input.size(1);
    AT_CHECK(input.dim() == 3, "not batch square input matrix");
    AT_CHECK(input.size(1) == n, "not batch square input matrix");
    AT_CHECK(input.size(2) == n, "not batch square input matrix");

    auto w = at::empty({batch_size, n}, input.type());
    auto v = at::empty({batch_size, n, n}, input.type());
    // FIXME: rewrite this with at::parallel_for in ATen/Parallel.h
#pragma omp for
    for (int64_t i = 0; i < batch_size; ++i)
    {
        at::Tensor wi, vi;
        // FIXME use syev directly to avoid fragmented memory alloc https://github.com/pytorch/pytorch/blob/695465915a88f4803dfae152151bb56be5c99410/aten/src/TH/generic/THat::TensorLapack.cpp#L361
        std::tie(wi, vi) = at::symeig(input.select(0, i), true, upper);
        w.select(0, i).copy_(wi);
        v.select(0, i).copy_(vi);
    }
//    return std::forward_as_tuple(w, v);
    return std::make_tuple(w, v);
}

at::Tensor batch_symeig_backward(
    // backward variables
    const at::Tensor& grad_loss_wrt_eigenvalues, // [b, m]
    const at::Tensor& grad_loss_wrt_eigenvectors, // [b, m, m]
    // forward variables
    const at::Tensor& x, // [b, m, m]
    const at::Tensor& eigenvalues, // [b, m]
    const at::Tensor& eigenvectors, // [b, m, m]
    // config
    bool upper)
{
    auto batch_size = x.size(0);
    auto m = x.size(1);

    AT_CHECK(x.dim() == 3, "not batch square input matrix");
    AT_CHECK(x.size(1) == m, "not batch square input matrix");
    AT_CHECK(x.size(2) == m, "not batch square input matrix");
    AT_CHECK(eigenvalues.dim() == 2, "invalid eigenvalues shape");
    AT_CHECK(eigenvalues.size(0) == batch_size, "invalid eigenvalues shape");
    AT_CHECK(eigenvalues.size(1) == m, "invalid eigenvalues shape");
    AT_CHECK(eigenvectors.dim() == 3, "invalid eigenvectors shape");
    AT_CHECK(eigenvectors.size(0) == batch_size, "invalid eigenvectors shape");
    AT_CHECK(eigenvectors.size(1) == m, "invalid eigenvectors shape");
    AT_CHECK(eigenvectors.size(2) == m, "invalid eigenvectors shape");

    AT_CHECK(grad_loss_wrt_eigenvalues.dim() == 2, "invalid grad_loss_wrt_eigenvalues shape");
    AT_CHECK(grad_loss_wrt_eigenvalues.size(0) == batch_size, "invalid grad_loss_wrt_eigenvalues shape");
    AT_CHECK(grad_loss_wrt_eigenvalues.size(1) == m, "invalid grad_loss_wrt_eigenvalues shape");
    AT_CHECK(grad_loss_wrt_eigenvectors.dim() == 3, "invalid grad_loss_wrt_eigenvectors shape");
    AT_CHECK(grad_loss_wrt_eigenvectors.size(0) == batch_size, "invalid grad_loss_wrt_eigenvectors shape");
    AT_CHECK(grad_loss_wrt_eigenvectors.size(1) == m, "invalid grad_loss_wrt_eigenvectors shape");
    AT_CHECK(grad_loss_wrt_eigenvectors.size(2) == m, "invalid grad_loss_wrt_eigenvectors shape");

    at::Tensor gx; // [b, m, m]
    auto vt = eigenvectors.transpose(1, 2);
    if (grad_loss_wrt_eigenvectors.defined())
    {
        auto F = eigenvalues.unsqueeze(1).expand_as(x).clone();
        F.sub_(at::unsqueeze(eigenvalues, 2));
        F.diagonal(0, 1, 2).fill_(INFINITY);
        F.reciprocal_();
        F.mul_(vt.bmm(grad_loss_wrt_eigenvectors));
        gx = eigenvectors.bmm(F.bmm(vt));
    }
    if (grad_loss_wrt_eigenvalues.defined())
    {
        auto gw_gx = (eigenvectors * grad_loss_wrt_eigenvalues.unsqueeze(-1)).bmm(vt);
        if (gx.defined()) {
            gx.add_(gw_gx);
        } else {
            gx = gw_gx;
        }
    }
    //
    if (upper)
        {
    //#pragma omp for
            int batch_batch_size= batch_size/65535;
            int left_batch_size= batch_size%65535;
            for (int64_t i = 0; i < batch_batch_size; ++i)
                {
        //         Tensorat::narrow(const Tensor &self, int64_t dim, int64_t start, int64_t length)
                    auto&& gxi = gx.narrow(0, 65535*(i-1),65535);
                    auto tmp_gxu =at::triu(gxi.transpose(1, 2), 1);
                    gxi=gxi.triu_().add_(tmp_gxu);
                }
            auto&& gxi2 = gx.narrow(0, 65535*batch_batch_size, left_batch_size);
            auto tmp_gxu2 =at::triu(gxi2.transpose(1, 2), 1);
            gxi2=gxi2.triu_().add_(tmp_gxu2);
        }
    else
        {
    //#pragma omp for
            auto tmp_gxu =at::tril(gx.transpose(1, 2), 1);
            gx=gx.tril_().add_(tmp_gxu);
        }



//          if (upper)
//     {
// //#pragma omp for
//         auto tmp_gxu =at::triu(gx.transpose(1, 2), 1);
//         gx=gx.triu_().add_(tmp_gxu);
//     }
//         else
//     {
// //#pragma omp for
//         auto tmp_gxu =at::tril(gx.transpose(1, 2), 1);
//         gx=gx.tril_().add_(tmp_gxu);
//     }



//    ori version
//    if (upper)
//    {
////       at::Tensor tmp_gx=gx.clone();
//        // FIXME: rewrite this with at::parallel_for in ATen/Parallel.h
//#pragma omp for
//        for (int64_t i = 0; i < batch_size; ++i)
//        {
//            auto&& gxi = gx.select(0, i);
//            auto gxu = at::triu(gxi.t(), 1);
//            gxi.triu_().add_(gxu);
//        }
//
////        auto tmp_gxu =at::triu(tmp_gx.transpose(1, 2), 1);
////        tmp_gx=tmp_gx.triu_().add_(tmp_gxu);
////        assert(tmp_gx != gx);
//    }
//    else
//    {
//#pragma omp for
//        for (int64_t i = 0; i < batch_size; ++i)
//        {
//            auto&& gxi = gx.select(0, i);
//            auto gxl = at::tril(gxi.t(), -1);
//            gxi.tril_().add_(gxl);
//        }
//    }


    return gx;
}



/**
   A
 */
//std::tuple<at::Tensor, at::Tensor> generalized_symeig_backward(
//    // backward variables
//    const at::Tensor& grad_loss_wrt_eigenvalues, // [m]
//    const at::Tensor& grad_loss_wrt_eigenvectors, // [m, m]
//    // forward variables
//    const at::Tensor& a, // [m, m]
//    const at::Tensor& b, // [m, m]
//    const at::Tensor& eigenvalues, // [m]
//    const at::Tensor& eigenvectors, // [m, m]
//    // config
//    bool upper)
//{
//    auto m = a.size(0);
//    AT_CHECK(a.dim() == 2, "not square input-A matrix");
//    AT_CHECK(a.size(1) == m, "not square input-A matrix");
//    AT_CHECK(b.dim() == 2, "not square input-B matrix");
//    AT_CHECK(b.size(0) == m, "not square input-B matrix");
//    AT_CHECK(b.size(1) == m, "not square input-B matrix");

//    AT_CHECK(eigenvalues.dim() == 1, "invalid eigenvalues shape");
//    AT_CHECK(eigenvalues.size(0) == m, "invalid eigenvalues shape");
//    AT_CHECK(eigenvectors.dim() == 2, "invalid eigenvectors shape");
//    AT_CHECK(eigenvectors.size(0) == m, "invalid eigenvectors shape");
//    AT_CHECK(eigenvectors.size(1) == m, "invalid eigenvectors shape");

//    AT_CHECK(grad_loss_wrt_eigenvalues.dim() == 1, "invalid grad_loss_wrt_eigenvalues shape");
//    AT_CHECK(grad_loss_wrt_eigenvalues.size(0) == m, "invalid grad_loss_wrt_eigenvalues shape");
//    AT_CHECK(grad_loss_wrt_eigenvectors.dim() == 2, "invalid grad_loss_wrt_eigenvectors shape");
//    AT_CHECK(grad_loss_wrt_eigenvectors.size(0) == m, "invalid grad_loss_wrt_eigenvectors shape");
//    AT_CHECK(grad_loss_wrt_eigenvectors.size(1) == m, "invalid grad_loss_wrt_eigenvectors shape");

//    auto ga = at::empty_like(a);
//    auto gb = at::empty_like(b);
//    return std::forward_as_tuple(a, b);
//}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("symeig_backward", &symeig_backward, "basic symeig backward");
    m.def("batch_symeig_forward", &batch_symeig_forward, "batch symeig forward");
    m.def("batch_symeig_backward", &batch_symeig_backward, "batch symeig backward");
//    m.def("batch_symeig_cuda_backward", &batch_symeig_cuda_backward, "batch cuda symeig backward");
//    m.def("generalized_symeig_backward", &generalized_symeig_backward, "generalized symeig backward");
}
