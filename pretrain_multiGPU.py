import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
import os
import glob
import wandb
from datasets import load_dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

from src.rwkv7 import RWKV7
from src.dataset import MyDataset
from src.transformer import TransformerModel

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def load_latest_checkpoint(model, checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoint_files:
        print("No checkpoint files found in the directory.")
        return 0
    latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded checkpoint: {checkpoint_path}")

def initialize_model(checkpoint_dir, dim, n_blocks, rank, world_size):
    # model = RWKV7(text_vocab=128, audio_vocab=8192 + 1, dim=dim, n_blocks=n_blocks).to(rank)
    model = TransformerModel(text_vocab=128, audio_vocab=8192 + 1, dim=dim, n_blocks=n_blocks).to(rank)
    
    if rank == 0:
        load_latest_checkpoint(model, checkpoint_dir)
    dist.barrier()
    model = DDP(model, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    
    # 从 rank 0 广播模型状态到所有其他 rank
    for param in model.parameters():
        dist.broadcast(param.data.clone(), src=0)  # 使用 clone() 方法
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model total parameters: {total_params}")
    print(f"Model trainable parameters: {trainable_params}")
    return model

def collate_fn(batch):
    padding_token = 8192
    max_length = max(len(seq) for seq in batch) - 1
    input_ids, targets, loss_masks = [], [], []
    for seq in batch:
        input_seq = list(seq[:-1])
        target_seq = list(seq[1:])
        input_padding = [padding_token] * (max_length - len(input_seq))
        target_padding = [padding_token] * (max_length - len(target_seq))
        mask_padding = [0] * (max_length - len(input_seq))
        input_ids.append(torch.tensor(input_seq + input_padding, dtype=torch.long))
        targets.append(torch.tensor(target_seq + target_padding, dtype=torch.long))
        loss_masks.append(torch.tensor([1] * len(input_seq) + mask_padding, dtype=torch.long))
    return torch.stack(input_ids, dim=0), torch.stack(targets, dim=0), torch.stack(loss_masks, dim=0)

def prepare_dataloader(batch_size, rank, world_size):
    dataset = load_dataset("JerryAGENDD/JLSpeech_tokenized", cache_dir="../temp_datasets")['train']
    dataset = MyDataset(hf_dataset=dataset, train_type='pretrain')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    return dataloader

def train(rank, world_size, args):
    setup(rank, world_size)
    model = initialize_model(args.checkpoint_dir, args.dim, args.n_blocks, rank,world_size)
    dataloader = prepare_dataloader(args.batch_size, rank, world_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if rank == 0:
        wandb.init(project="TTS")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model.train()
    for epoch in range(args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        if rank == 0:
            epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=False)
        else:
            epoch_iterator = dataloader
        
        for batch in epoch_iterator:
            input_ids, targets, loss_masks = batch
            input_ids = input_ids.long().to(rank)
            targets = targets.long().to(rank)
            loss_masks = loss_masks.to(rank)
            outputs = model(None, None, input_ids)
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss = loss.view(targets.size()) * loss_masks
            loss = loss.sum() / loss_masks.sum()
            
            if rank == 0:
                wandb.log({"loss": loss.item()})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if rank == 0:
            save_checkpoint(model, args.checkpoint_dir, epoch)
    
    if rank == 0:
        wandb.finish()
    
    cleanup()

def save_checkpoint(model, output_dir, epoch):
    pt_files = glob.glob(os.path.join(output_dir, "*.pt"))
    for pt_file in pt_files:
        os.remove(pt_file)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    torch.save(model.module.state_dict(), checkpoint_path)

def main():
    parser = argparse.ArgumentParser(description="Train RWKV7 model")
    parser.add_argument("--dim", type=int, default=128, help="Dimension of the model")
    parser.add_argument("--n_blocks", type=int, default=5, help="Number of blocks in the model")
    parser.add_argument("--num_epochs", type=int, default=4000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()