import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import glob
import wandb
from datasets import load_dataset
from accelerate import Accelerator
import argparse

from src.rwkv7 import RWKV7
from src.dataset import MyDataset

def load_latest_checkpoint(model, checkpoint_dir):
    """
    Load the latest checkpoint for the model from the specified directory.
    
    Args:
        model: The model to load the checkpoint into
        checkpoint_dir: Directory containing checkpoint files (.pt)
    """
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        print("No checkpoint files found in the directory.")
        return 0
    latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint: {checkpoint_path}")

def initialize_model(checkpoint_dir, dim, n_blocks):
    """
    Initialize the RWKV7 model and load the latest checkpoint.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        dim: Dimension of the model
        n_blocks: Number of blocks in the model
    
    Returns:
        The initialized model
    """
    # Initialize model
    model = RWKV7(text_vocab=128, audio_vocab=8192 + 1, dim=dim, n_blocks=n_blocks).cuda()
    
    # Load latest checkpoint
    load_latest_checkpoint(model, checkpoint_dir)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model total parameters: {total_params}")
    print(f"Model trainable parameters: {trainable_params}")
    
    return model

def collate_fn(batch):
    # 分离文本和音频token id列表
    text_token_ids = [item[0] for item in batch]
    audio_token_ids = [item[1] for item in batch]

    # 确定填充的最大长度
    max_text_len = max(len(text) for text in text_token_ids)
    max_audio_len = max(len(audio) for audio in audio_token_ids) + 1  # +1用于结束符

    # 初始化填充后的列表和掩码
    text_input_ids = []
    text_attention_mask = []
    audio_input_ids = []
    target = []
    loss_mask = []

    # 填充值
    text_pad_token = 38
    audio_pad_token = 8192
    target_pad_token = 8192

    for text, audio in zip(text_token_ids, audio_token_ids):
        # 填充文本输入id并创建注意力掩码
        padded_text = text + [text_pad_token] * (max_text_len - len(text))
        attention_mask = [1] * len(text) + [0] * (max_text_len - len(text))

        # 填充音频输入id并添加结束符
        padded_audio = audio + [audio_pad_token] * (max_audio_len - len(audio) - 1)

        # 创建目标和损失掩码
        target_seq = [0] * (len(text) - 1) + audio + [target_pad_token] * (max_audio_len + max_text_len - len(text) - len(audio))
        loss_mask_seq = [0] * (len(text) - 1) + [1] * (len(audio) + 1) + [0] * (max_audio_len + max_text_len - len(text) - len(audio) - 1)

        # 添加到列表中
        text_input_ids.append(padded_text)
        text_attention_mask.append(attention_mask)
        audio_input_ids.append(padded_audio)
        target.append(target_seq)
        loss_mask.append(loss_mask_seq)

    return torch.tensor(text_input_ids), torch.tensor(text_attention_mask), torch.tensor(audio_input_ids), torch.tensor(target), torch.tensor(loss_mask)

def prepare_dataloader(batch_size):
    """
    Prepare dataset and dataloader.
    
    Args:
        batch_size: Batch size for training
        
    Returns:
        DataLoader for training
    """
    # Load dataset
    dataset = load_dataset("JerryAGENDD/JLSpeech_tokenized", cache_dir="../temp_datasets")['train']
    dataset = MyDataset(hf_dataset=dataset, train_type='sft')
   
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    return dataloader

def train(model, dataloader, num_epochs, output_dir, learning_rate):
    """
    Train the model.
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        num_epochs: Number of training epochs
        output_dir: Directory to save checkpoints
        learning_rate: Learning rate for optimizer
    """
    # Set up accelerator and optimizer
    accelerator = Accelerator()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)
    
    # Initialize wandb
    wandb.init(project="TTS")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for batch in tqdm(dataloader, leave=False):
            text_input_ids, text_attention_mask, audio_input_ids, targets, loss_masks = batch
            
            text_input_ids = text_input_ids.long().to('cuda')
            text_attention_mask = text_attention_mask.to('cuda')
            audio_input_ids = audio_input_ids.long().to('cuda')
            targets = targets.long().to('cuda')
            loss_masks = loss_masks.to('cuda')

            # Forward pass
            outputs = model(text_input_ids, text_attention_mask, audio_input_ids)

            # Calculate loss
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            # Apply loss masks
            loss = loss.view(targets.size()) * loss_masks
            loss = loss.sum() / loss_masks.sum()  # Calculate average loss

            # Log to wandb
            wandb.log({"loss": loss.item()})
            
            # Backward pass and optimization
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        # Save checkpoint at the end of each epoch
        save_checkpoint(model, output_dir, epoch)
    
    # Finish the wandb run
    wandb.finish()

def save_checkpoint(model, output_dir, epoch):
    """
    Save a model checkpoint.
    
    Args:
        model: The model to save
        output_dir: Directory to save the checkpoint
        epoch: Current epoch number
    """
    # Delete all existing checkpoint files
    pt_files = glob.glob(os.path.join(output_dir, "*.pt"))
    for pt_file in pt_files:
        os.remove(pt_file)

    # Save current checkpoint
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    # print(f"Saved checkpoint to {checkpoint_path}")

def test():
    batch = [
    ([1, 2, 3, 38], [100, 101, 102]),
    ([4, 5, 38], [103, 104, 105]),
    ([6, 7, 8, 9, 38], [106, 107, 108, 109, 110])
    ]
    
    text_input_ids, text_attention_mask, audio_input_ids, target, loss_mask = collate_fn(batch)
    
    print(f"text_input_ids:{text_input_ids}")
    print(f"text_attention_mask:{text_attention_mask}")
    print(f"audio_input_ids:{audio_input_ids}")
    print(f"target:{target}")
    print(f"loss_mask:{loss_mask}")
    
    model = initialize_model("./checkpoints")
    output = model(text_input_ids.long().to('cuda'), text_attention_mask.to('cuda'), audio_input_ids.long().to('cuda'))
    return 

def main():
    """
    Main function to run the training process.
    """
    parser = argparse.ArgumentParser(description="Train RWKV7 model")
    parser.add_argument("--dim", type=int, default=128, help="Dimension of the model")
    parser.add_argument("--n_blocks", type=int, default=5, help="Number of blocks in the model")
    parser.add_argument("--num_epochs", type=int, default=4000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    args = parser.parse_args()

    # Configuration
    checkpoint_dir = "./checkpoints"
    
    # Initialize model
    model = initialize_model(checkpoint_dir, args.dim, args.n_blocks)
    
    # Prepare dataloader
    dataloader = prepare_dataloader(args.batch_size)
    # Train model
    train(model, dataloader, args.num_epochs, checkpoint_dir, args.learning_rate)

if __name__ == "__main__":
    main()