import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.cpp_extension import load

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script

D_DECAY_LORA = 64
D_AAA_LORA = 64
D_MV_LORA = 32
D_GATE_LORA = 128
HEAD_SIZE = 64
DTYPE = torch.bfloat16

CHUNK_LEN = 16

flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="wind_backstepping", sources=[f'src/cuda/wkv7_cuda.cu', 'src/cuda/wkv7_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0, f"w must be divisible by CHUNK_LEN = {CHUNK_LEN}, got {T}"
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b]), "w,q,k,v,z,b must be bfloat16"
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b]), "w,q,k,v,z,b must be contiguous"
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy]), "dy must be bfloat16"
        assert all(i.is_contiguous() for i in [dy]), "dy must be contiguous"
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

class RWKV_Tmix_x070(MyModule):
    def __init__(self, dim, blocks, layer_id):
        super().__init__()
        # self.args = args
        self.layer_id = layer_id
        # self.my_testing = args.my_testing

        self.head_size = HEAD_SIZE
        self.n_head = dim // self.head_size
        assert dim % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = dim

        #初始化
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (blocks - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / blocks)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()

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

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    

class RWKV_CMix_x070(MyModule):
    def __init__(self, dim, ffn_dim,blocks, layer_id):
        super().__init__()
        # self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / blocks)  # 1 to ~0
            ddd = torch.ones(1, 1, dim)
            for i in range(dim):
                ddd[0, 0, i] = i / dim
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(dim, ffn_dim, bias=False)
        self.value = nn.Linear(ffn_dim, dim, bias=False)

        #初始化
        self.key.weight.data.uniform_(-0.5/(dim**0.5), 0.5/(dim**0.5))
        self.value.weight.data.zero_()
        
    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

class Block(MyModule):
    def __init__(self,dim, blocks, layer_id):
        super().__init__()
        # self.args = args
        self.layer_id = layer_id

         # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        self.att = RWKV_Tmix_x070(dim, blocks, layer_id)
        self.ffn = RWKV_CMix_x070(dim,dim*4, blocks, layer_id)
        
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
        self.blocks = nn.ModuleList([Block(dim,n_blocks, i) for i in range(n_blocks)])

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