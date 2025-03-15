import torch
from src.rwkv7 import RWKV7
import os

# 初始化模型并移动到GPU
model = RWKV7(text_vocab=128, audio_vocab=8192 + 1, dim=128, n_blocks=5).cuda()

def load_latest_checkpoint(model, checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        print("No checkpoint files found in the directory.")
        return 0
    latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint: {checkpoint_path}")

checkpoint_dir = "./checkpoints"
load_latest_checkpoint(model, checkpoint_dir)

config = "lucadellalib/focalcodec_12_5hz"
codec = torch.hub.load("lucadellalib/focalcodec", "focalcodec", config=config, force_reload=False)
codec.eval().requires_grad_(False)
codec.to('cuda')
##############################################################################################################


from tqdm import tqdm
from datasets import load_dataset

ds = load_dataset("JerryAGENDD/JLSpeech_tokenized", cache_dir="../temp_datasets")['train'][0]

LEN = len(ds['audio_token'])
print(ds['normalized_text'])
tokens = ds['audio_token'][:10]
tokens_tensor = torch.tensor(tokens).unsqueeze(0).cuda()  # 将tokens转换为tensor并移动到CUDA

for le in tqdm(range(10, LEN)):
    outputs = model(None, tokens_tensor)
    # 获取最后一个logit
    last_logit = outputs[0, -1, :]
    # 对最后一个logit进行最大采样
    next_token = torch.argmax(last_logit).item()
    # 将采样后的token添加到tokens列表中
    tokens.append(next_token)
    # 更新tokens_tensor
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).cuda()


target = ds['audio_token']
# 将target转换为tensor并移动到CUDA设备
target_tensor = torch.tensor(target).cuda().unsqueeze(0)

# 将tokens转换为tensor并移动到CUDA设备
tokens_tensor = torch.tensor(tokens).cuda().unsqueeze(0)

print(target_tensor.shape)
print(tokens_tensor.shape)
# 使用codec进行解码
target_signal = codec.toks_to_sig(target_tensor).squeeze(0)
# 使用codec进行解码
example_signal = codec.toks_to_sig(tokens_tensor).squeeze(0)


import scipy.io.wavfile as wavfile
# 将tensor转换回list
target_signal_list = target_signal.cpu().numpy()
example_signal_list = example_signal.cpu().numpy()

# 写入wav文件
wavfile.write('output_target.wav', codec.sample_rate, target_signal_list)
wavfile.write('output_model.wav', codec.sample_rate, example_signal_list)
