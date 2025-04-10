#include <torch/extension.h>
#include <assert.h> // 修改：移除 <cuda_fp16.h>，添加 <assert.h>
using fp = float; // 修改：将 bf 替换为 fp，并将 __nv_bfloat16 替换为 float

void cuda_forward(int B, int T, int H, fp*w, fp*q, fp*k, fp*v, fp*z, fp*a, fp*y, float*s, float*sa); // 修改：将 bf* 替换为 fp*

void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, (fp*)w.data_ptr(), (fp*)q.data_ptr(), (fp*)k.data_ptr(), (fp*)v.data_ptr(), (fp*)z.data_ptr(), (fp*)a.data_ptr(), (fp*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr()); // 修改：将 (bf*) 替换为 (fp*)
}

void cuda_backward(int B, int T, int H, fp*w, fp*q, fp*k, fp*v, fp*z, fp*a, fp*dy, float*s, float*sa, fp*dw, fp*dq, fp*dk, fp*dv, fp*dz, fp*da); // 修改：将 bf* 替换为 fp*

void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, torch::Tensor &dy,
        torch::Tensor &s, torch::Tensor &sa, torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, (fp*)w.data_ptr(), (fp*)q.data_ptr(), (fp*)k.data_ptr(), (fp*)v.data_ptr(), (fp*)z.data_ptr(), (fp*)a.data_ptr(), (fp*)dy.data_ptr(), 
            (float*)s.data_ptr(), (float*)sa.data_ptr(), (fp*)dw.data_ptr(), (fp*)dq.data_ptr(), (fp*)dk.data_ptr(), (fp*)dv.data_ptr(), (fp*)dz.data_ptr(), (fp*)da.data_ptr()); // 修改：将 (bf*) 替换为 (fp*)
}

TORCH_LIBRARY(wind_backstepping, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da) -> ()");
}

TORCH_LIBRARY_IMPL(wind_backstepping, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}