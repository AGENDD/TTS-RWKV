import torch
import torch.nn.functional as F
from torch import nn

from fla.layers import RWKV7Attention  # type: ignore
from fla.utils import device

class TMix(nn.Module):
    def __init__(self, dim, block_id, n_blocks):
        super().__init__()
        self.rwkv7 = RWKV7Attention(
            "chunk",
            dim,
            layer_idx=block_id
        )

    def forward(self, x, v_first):
        x_attn, _, past_key_values, v_first = self.rwkv7(x, v_first=v_first)
        return x_attn, v_first

class CMix(nn.Module):
    def __init__(self, dim, hidden_dim, block_id, n_blocks):
        super().__init__()
        self.value1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.value2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        x = self.relu(self.value1(x))
        return self.value2(x)

class RWKV7Block(nn.Module):
    def __init__(self, dim, block_id, n_blocks):
        super().__init__()
        self.attn = TMix(dim, block_id, n_blocks)
        self.mlp = CMix(dim, dim*2, block_id, n_blocks)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, v_first):
        x_attn, v_first = self.attn(self.norm1(x), v_first=v_first)
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x, v_first

class RWKV7(nn.Module):
    def __init__(self,
            text_vocab,     # vocab size of text tokens
            audio_vocab,    # vocab size of audio tokens, also the output token
            dim, 
            n_blocks: int):
        super().__init__()
        
        self.text_embed = nn.Embedding(text_vocab, dim)
        self.audio_embed = nn.Embedding(audio_vocab, dim)
        
        self.blocks = nn.ModuleList([
            RWKV7Block(dim, i, n_blocks)
            for i in range(n_blocks)
        ])
        self.lmhead = nn.Linear(dim, audio_vocab)
        self.norm_in = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, text, audio):
        if(text is None and audio is not None):         #pretrain
            audio_emb = self.audio_embed(audio)
            x = audio_emb
        elif(text is not None and audio is not None):   #sft or inference
            audio_emb = self.audio_embed(audio)
            text_emb = self.text_embed(text)
            # concat
            x = torch.cat((text, audio), dim=1)
        elif(text is not None and audio is None):       #inference
            text_emb = self.text_embed(text)
            x = text_emb

        
        x = self.norm_in(x)
        v_first = None
        for block in self.blocks:
            x, v_first = block(x, v_first)
            
        return self.lmhead(self.norm_out(x))