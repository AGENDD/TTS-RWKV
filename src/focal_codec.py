import torch
import torchaudio
"""
加载模型
"""
# Load FocalCodec model
config = "lucadellalib/focalcodec_12_5hz"
codec = torch.hub.load(
    "lucadellalib/focalcodec", "focalcodec", config=config, force_reload=False
)
codec.eval().requires_grad_(False)

"""
读取音频
"""
# Load and preprocess the input audio
audio_file = "/Users/kongjiaming/Documents/1721129273595_16k.wav"
sig, sample_rate = torchaudio.load(audio_file)
sig = torchaudio.functional.resample(sig, sample_rate, codec.sample_rate)

"""
将音频的波形信号编码成semantic tokens
"""
# Encode audio into tokens
toks = codec.sig_to_toks(sig)  # Shape: (batch, time)
print(f"Token tensor shape: {toks.shape}")
print(f"Token tensor values:\n{toks}")

import pdb; pdb.set_trace()
"""
这里只是用来看semantic tokens对应码本里面的值，在我们实际训练里并不需要这些内容
"""
codes = codec.toks_to_codes(toks)  # Shape: (batch, time, log2 codebook_size)
print(f"Code tensor shape: {codes.shape}")
print(f"Code tensor values:\n{codes}")


"""
使用semantic tokens重建音频。
"""

# Decode tokens back into a waveform
rec_sig = codec.toks_to_sig(toks)
rec_sig = torchaudio.functional.resample(rec_sig, codec.sample_rate, sample_rate)
torchaudio.save("reconstruction.wav", rec_sig, sample_rate)