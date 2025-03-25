import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.ReLU(),
            nn.Linear(2*dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_output
        x = self.norm2(x)
        ff_output = self.ff(x)
        x = x + ff_output
        return x

class TransformerModel(nn.Module):
    def __init__(self, text_vocab, audio_vocab, dim, n_blocks, n_heads = 8):
        super().__init__()
        self.text_embed = nn.Embedding(text_vocab, dim)
        self.audio_embed = nn.Embedding(audio_vocab, dim)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads)
            for _ in range(n_blocks)
        ])
        
        self.pos_encoder = PositionalEncoding(dim)
        self.lmhead = nn.Linear(dim, audio_vocab)
        self.norm_in = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, text, text_attention_mask, audio):
        
        if text is None and audio is not None:  # pretrain
            audio = self.audio_embed(audio)
            x = audio
        elif text is not None and audio is not None:  # sft or inference
            text = self.text_embed(text)
            audio = self.audio_embed(audio)

            
            text_list = []
            audio_list = []
            
            for i in range(text.shape[0]):
                masked_text = text[i][text_attention_mask[i] == 1]
                text_list.append(masked_text)
                audio_segment = audio[i]
                audio_list.append(audio_segment)
                
            tensor_list = []
            max_len = max([len(x) + len(y) for x, y in zip(text_list, audio_list)])
            
            for i in range(len(text_list)):
                combine_tensor = torch.cat((text_list[i], audio_list[i]), dim=0)
                if combine_tensor.shape[0] < max_len:
                    padding = audio_list[i][-1:].expand(max_len - combine_tensor.shape[0], -1)
                    combine_tensor = torch.cat((combine_tensor, padding), dim=0)
                tensor_list.append(combine_tensor)
                    
            x = torch.stack(tensor_list, dim=0)
            
        elif text is not None and audio is None:  # inference
            x = self.text_embed(text)
            
        x = self.pos_encoder(x)
        
        x = self.norm_in(x)
        for block in self.blocks:
            x = block(x)
            
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

            x = self.pos_encoder(x)
            x = self.norm_in(x)
            for block in self.blocks:
                x = block(x)
            x = self.lmhead(self.norm_out(x))
            
            last_vector = x[:, -1, :]
            probabilities = F.softmax(last_vector / temperature, dim=-1)
            token_id = torch.argmax(probabilities, dim=-1)
            print(token_id.item())
            
            if token_id.item() == 8192:
                print("ending with pad")
                return torch.tensor(tokens).unsqueeze(0).to(device)

            tokens.append(token_id.item())
            audio = torch.tensor(tokens).unsqueeze(0).to(device)

        return torch.tensor(tokens).unsqueeze(0).to(device)