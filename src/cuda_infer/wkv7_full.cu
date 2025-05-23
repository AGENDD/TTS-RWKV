#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _w, const F *__restrict__ const _k, const F *__restrict__ const _v, const F *__restrict__ const _a, const F *__restrict__ const _b,
                               F *__restrict__ const _y)
{
    const int e = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    float state[_N_] = {0};
    __shared__ float r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];

    for (int _t = 0; _t < T; _t++)
    {
        const int t = e*T*C + h*_N_ + i + _t * C;
        __syncthreads();
        r[i] = _r[t];
        w[i] = __expf(-__expf(_w[t]));
        k[i] = _k[t];
        a[i] = _a[t];
        b[i] = _b[t];
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            sa += a[j] * state[j];
        }

        float vv = _v[t];
        float y = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            s = s * w[j] + k[j] * vv + sa * b[j];
            y += s * r[j];
        }
        _y[t] = y;
    }
}

void cuda_forward(int B, int T, int C, int H, float *r, float* w, float *k, float *v, float *a, float *b, float *y)
{
    assert(H*_N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, w, k, v, a, b, y);
}
