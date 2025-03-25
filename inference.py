import os
from src.rwkv7 import RWKV7
from src.text_tokenizer import TextTokenizer
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:3")

model = RWKV7(text_vocab=128, audio_vocab=8192 + 1, dim=256, n_blocks=12).to(device)
tokenizer = TextTokenizer()
config = "lucadellalib/focalcodec_12_5hz"
codec = torch.hub.load(
    "lucadellalib/focalcodec", "focalcodec", config=config, force_reload=False
)
codec.eval().requires_grad_(False).to(device)

checkpoint_dir = './checkpoints'
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
if not checkpoint_files:
    print("No checkpoint files found in the directory.")
    exit(0)
latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
model.load_state_dict(torch.load(checkpoint_path))
print(f"Loaded checkpoint: {checkpoint_path}")

MAX_LENGTH = 2000
print("Start")
while(True):
    text = str(input())
    print("Computing...")
    tokens = tokenizer.tokenize(text)
    text_tensor = torch.tensor(tokens).unsqueeze(0).to(device) # 1, seq, 
    print(f"text_tensor:{text_tensor}")
    tokens = model.generate(None, text_tensor, 2000, device) #return a tensor
    # print(tokens.shape)
    print(tokens)
    signal = codec.toks_to_sig(tokens).squeeze(0)

    import scipy.io.wavfile as wavfile
    signal_list = signal.cpu().numpy()
    wavfile.write(f'test.wav', codec.sample_rate, signal_list)
    
    print("Finish")
        
        