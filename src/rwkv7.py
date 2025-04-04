import torch
import torch.nn.functional as F
from torch import nn

from fla.layers import RWKV7Attention  # type: ignore
from fla.utils import device
import torch.nn.init as init

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
        self.dropout = nn.Dropout(p=0.3)
        self.blocks = nn.ModuleList([
            RWKV7Block(dim, i, n_blocks)
            for i in range(n_blocks)
        ])
        self.lmhead = nn.Linear(dim, audio_vocab)
        self.norm_in = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, text,text_attention_mask, audio):
        if(text is None and audio is not None):         #pretrain
            audio = self.audio_embed(audio)
            x = audio
        elif(text is not None and audio is not None):   #sft or inference
            text = self.text_embed(text)
            audio = self.audio_embed(audio)
            
            text_list = []
            audio_list = []
            
            # 接下来这一步把text embed 和 audio embed 进行拼接。再次之前要把text embed右边的padding挖掉，凭借后在再audio embed的右边进行补全
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
            
        # x = self.dropout(x)
        x = self.norm_in(x)
        v_first = None
        for block in self.blocks:
            x, v_first = block(x, v_first)
            
        return self.lmhead(self.norm_out(x))
    
    def generate(self, audio, text, MAX_LENGTH, device, temperature=1.0):

        tokens = []
        for i in range(MAX_LENGTH):
            text_out = self.text_embed(text)
            if audio is None:
                x = text_out
            else:
                audio_out = self.audio_embed(audio)
                x = torch.cat([text_out, audio_out], dim=1)

            x = self.norm_in(x)
            v_first = None
            for block in self.blocks:
                x, v_first = block(x, v_first)
            x = self.lmhead(self.norm_out(x))
            # print(x.shape)
            
            last_vector = x[:, -1, :]
            probabilities = F.softmax(last_vector / temperature, dim=-1)
            token_id = torch.argmax(probabilities, dim=-1)
            print(token_id.item())
            
            if token_id.item() == 8192:
                print("ending with pad")
                return torch.tensor(tokens).unsqueeze(0).to(device)

            tokens.append(token_id.item())  # 转换为整数
            audio = torch.tensor(tokens).unsqueeze(0).to(device)

        return torch.tensor(tokens).unsqueeze(0).to(device)
    
    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                init.normal_(module.weight, mean=0, std=0.01)
            elif isinstance(module, nn.LayerNorm):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)