import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

# args.n_layer = 12
# args.n_embd = 768
D_DECAY_LORA = 64
D_AAA_LORA = 64
D_MV_LORA = 32
D_GATE_LORA = 128
HEAD_SIZE = 64

DTYPE = torch.float32
# DTYPE = torch.bfloat16

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script

from torch.utils.cpp_extension import load

load(name="wkv7", sources=["src/cuda_infer/wkv7_op_full.cpp", f"src/cuda_infer/wkv7_full.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])

class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H, f"HEAD_SIZE ({HEAD_SIZE}) must be equal to C // H ({C // H})"
            assert r.dtype == DTYPE, f"r.dtype ({r.dtype}) must be {DTYPE}"
            assert w.dtype == DTYPE, f"w.dtype ({w.dtype}) must be {DTYPE}"
            assert k.dtype == DTYPE, f"k.dtype ({k.dtype}) must be {DTYPE}"
            assert v.dtype == DTYPE, f"v.dtype ({v.dtype}) must be {DTYPE}"
            assert a.dtype == DTYPE, f"a.dtype ({a.dtype}) must be {DTYPE}"
            assert b.dtype == DTYPE, f"b.dtype ({b.dtype}) must be {DTYPE}"
            assert r.is_contiguous(), "r must be contiguous"
            assert w.is_contiguous(), "w must be contiguous"
            assert k.is_contiguous(), "k must be contiguous"
            assert v.is_contiguous(), "v must be contiguous"
            assert a.is_contiguous(), "a must be contiguous"
            assert b.is_contiguous(), "b must be contiguous"
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
            torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
            return y

def RWKV7_OP(r, w, k, v, a, b):
    return WKV_7.apply(r, w, k, v, a, b)


class RWKV_Tmix_x070(MyModule):
    def __init__(self, dim, layer_id):
        super().__init__()
        # self.args = args
        self.layer_id = layer_id

        self.head_size = HEAD_SIZE
        self.n_head = dim // self.head_size
        assert dim % self.n_head == 0

        H = self.n_head
        N = self.head_size
        C = dim

        self.x_r = nn.Parameter(torch.empty(1,1,C))
        self.x_w = nn.Parameter(torch.empty(1,1,C))
        self.x_k = nn.Parameter(torch.empty(1,1,C))
        self.x_v = nn.Parameter(torch.empty(1,1,C))
        self.x_a = nn.Parameter(torch.empty(1,1,C))
        self.x_g = nn.Parameter(torch.empty(1,1,C))

        self.w0 = nn.Parameter(torch.empty(1,1,C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.empty(1,1,C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        self.v0 = nn.Parameter(torch.empty(1,1,C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        self.k_k = nn.Parameter(torch.empty(1,1,C))
        self.k_a = nn.Parameter(torch.empty(1,1,C))
        self.r_k = nn.Parameter(torch.empty(H,N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first

class RWKV_CMix_x070(MyModule):
    def __init__(self, dim, dim_ffn, layer_id):
        super().__init__()
        # self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, dim))

        self.key = nn.Linear(dim, dim_ffn, bias=False)
        self.value = nn.Linear(dim_ffn, dim, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

class Block(MyModule):
    def __init__(self, dim, layer_id):
        super().__init__()
        # self.args = args
        self.layer_id = layer_id

        # self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.att = RWKV_Tmix_x070(dim, layer_id)
        self.ffn = RWKV_CMix_x070(dim, dim*4, layer_id)
        
    @MyFunction
    def forward(self, x, v_first):

        # if self.layer_id == 0:
        #     x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))

        return x, v_first


class RWKV7(nn.Module):
    def __init__(self,
            text_vocab,     # vocab size of text tokens
            audio_vocab,    # vocab size of audio tokens, also the output token
            dim, 
            n_blocks: int):
        super().__init__()
        # args.dim_att = dim
        # args.dim_ffn = dim * 4
        
        
        
        self.text_embed = nn.Embedding(text_vocab, dim)
        self.audio_embed = nn.Embedding(audio_vocab, dim)
        
        
        self.ln_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([Block(dim, i) for i in range(n_blocks)])

        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, audio_vocab, bias=False)

    def forward(self,  text,text_attention_mask, audio):
        
        if(text is None and audio is not None):         #pretrain
            audio = self.audio_embed(audio)
            x = audio
        elif(text is not None and audio is not None):   #sft or inference
            text = self.text_embed(text)
            audio = self.audio_embed(audio)
            
            text_list = []
            audio_list = []
            
            # 接下来这一步把text embed 和 audio embed 进行拼接。在此之前要把text embed右边的padding挖掉，拼接后在再audio embed的右边进行补全
            # Iterate over the batch dimension
            for i in range(text.shape[0]):
                # Mask the text based on text_attention_mask
                masked_text = text[i][text_attention_mask[i] == 1]
                text_list.append(masked_text)
                # Get the corresponding audio segment
                audio_segment = audio[i]
                audio_list.append(audio_segment)
                
            tensor_list = []
            max_len = max([len(x)+len(y) for x, y in zip(text_list, audio_list)])
            
            for i in range(len(text_list)):
                combine_tensor = torch.cat((text_list[i], audio_list[i]), dim=0)
                if combine_tensor.shape[0] < max_len:
                    padding = audio_list[i][-1:].expand(max_len - combine_tensor.shape[0], -1)
                    combine_tensor = torch.cat((combine_tensor, padding), dim=0)
                tensor_list.append(combine_tensor)
                    
            x = torch.stack(tensor_list,dim=0)
            
        elif(text is not None and audio is None):       #不存在这个情况
            x = self.text_embed(text)
        
        x = self.ln_in(x)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)

        return x