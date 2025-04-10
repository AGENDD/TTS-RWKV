#include <torch/extension.h>
#include "ATen/ATen.h"

// 前向声明
void cuda_forward(int B, int T, int C, int H, float *r, float *w, float *k, float *v, float *a, float *b, float *y);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, 
            torch::Tensor &r,  // float32 tensor
            torch::Tensor &w,  // float32 tensor
            torch::Tensor &k,  // float32 tensor
            torch::Tensor &v,  // float32 tensor
            torch::Tensor &a,  // float32 tensor
            torch::Tensor &b,  // float32 tensor
            torch::Tensor &y)  // float32 tensor
{
    cuda_forward(B, T, C, H, 
                r.data_ptr<float>(), 
                w.data_ptr<float>(), 
                k.data_ptr<float>(), 
                v.data_ptr<float>(), 
                a.data_ptr<float>(), 
                b.data_ptr<float>(), 
                y.data_ptr<float>());
}

TORCH_LIBRARY(wkv7, m) {
    m.def("forward", forward);
}
