import os
import torch
import scipy.io.wavfile as wavfile
from tqdm import tqdm
from datasets import load_dataset
from src.rwkv7 import RWKV7

def load_latest_checkpoint(model, checkpoint_dir):
    """
    Load the latest checkpoint for the model from the specified directory.
    
    Args:
        model: The model to load the checkpoint into
        checkpoint_dir: Directory containing checkpoint files (.pt)
        
    Returns:
        int: 0 if no checkpoint was found
    """
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        print("No checkpoint files found in the directory.")
        return 0
    
    latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint: {checkpoint_path}")

def initialize_model():
    """
    Initialize the RWKV7 model and load the latest checkpoint.
    
    Returns:
        The initialized model
    """
    # Initialize model and move to GPU
    model = RWKV7(text_vocab=128, audio_vocab=8192 + 1, dim=128, n_blocks=5).cuda()
    
    # Load latest checkpoint
    checkpoint_dir = "./checkpoints"
    load_latest_checkpoint(model, checkpoint_dir)
    
    return model

def load_codec():
    """
    Load and initialize the audio codec model.
    
    Returns:
        The initialized codec model
    """
    config = "lucadellalib/focalcodec_12_5hz"
    codec = torch.hub.load("lucadellalib/focalcodec", "focalcodec", config=config, force_reload=False)
    codec.eval().requires_grad_(False)
    codec.to('cuda')
    
    return codec

def load_dataset_sample():
    """
    Load a sample from the JLSpeech dataset.
    
    Returns:
        A sample from the dataset
    """
    ds = load_dataset("JerryAGENDD/JLSpeech_tokenized", cache_dir="../temp_datasets")['train'][0]
    print(f"Loaded sample with text: {ds['normalized_text']}")
    return ds

def generate_audio_tokens(model, initial_tokens, target_length):
    """
    Generate audio tokens using the model based on initial tokens.
    
    Args:
        model: The RWKV7 model
        initial_tokens: List of initial tokens to start generation from
        target_length: Total length of tokens to generate
    
    Returns:
        List of generated tokens
    """
    tokens = initial_tokens.copy()  # Create a copy to avoid modifying the original list
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).cuda()
    
    for _ in tqdm(range(len(initial_tokens), target_length)):
        outputs = model(None, tokens_tensor)
        # Get the last logit
        last_logit = outputs[0, -1, :]
        # Sample from the logits (using argmax for deterministic output)
        next_token = torch.argmax(last_logit).item()
        # Add the sampled token
        tokens.append(next_token)
        # Update tokens tensor
        tokens_tensor = torch.tensor(tokens).unsqueeze(0).cuda()
    
    return tokens

def save_audio_file(signal, filename, sample_rate):
    """
    Save audio signal to a WAV file.
    
    Args:
        signal: Audio signal tensor
        filename: Output filename
        sample_rate: Sample rate of the audio
    """
    signal_array = signal.cpu().numpy()
    wavfile.write(filename, sample_rate, signal_array)
    print(f"Saved audio to {filename}")

def main():
    # Initialize model and codec
    model = initialize_model()
    codec = load_codec()
    
    # Load dataset
    ds = load_dataset_sample()
    
    # Get target tokens and initial tokens for generation
    target_tokens = ds['audio_token']
    initial_tokens = target_tokens[:10]
    
    # Generate audio tokens
    print("Generating audio tokens...")
    generated_tokens = generate_audio_tokens(model, initial_tokens, len(target_tokens))
    
    # Convert tokens to tensors for decoding
    target_tensor = torch.tensor(target_tokens).unsqueeze(0).cuda()
    generated_tensor = torch.tensor(generated_tokens).unsqueeze(0).cuda()
    
    print(f"Target shape: {target_tensor.shape}")
    print(f"Generated shape: {generated_tensor.shape}")
    
    # Decode audio signals
    print("Decoding audio signals...")
    target_signal = codec.toks_to_sig(target_tensor).squeeze(0)
    generated_signal = codec.toks_to_sig(generated_tensor).squeeze(0)
    
    # Save audio files
    save_audio_file(target_signal, 'output_target.wav', codec.sample_rate)
    save_audio_file(generated_signal, 'output_model.wav', codec.sample_rate)

if __name__ == "__main__":
    main()
