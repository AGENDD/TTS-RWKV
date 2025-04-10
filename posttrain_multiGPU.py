import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
import os
import glob
import wandb
from datasets import load_dataset, concatenate_datasets
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse

from src.rwkv7 import RWKV7
from src.dataset import MyDataset

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return grad_output, gy

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
    model = RWKV7(text_vocab=128, audio_vocab=8192 + 1, dim=dim, n_blocks=n_blocks).to(rank)
    # model = TransformerModel(text_vocab=128, audio_vocab=8192 + 1, dim=dim, n_blocks=n_blocks).to(rank)
    
    if rank == 0:
        load_latest_checkpoint(model, checkpoint_dir)
    dist.barrier()
    model = DDP(model, device_ids=[rank],find_unused_parameters=True, broadcast_buffers=False)
    
    # 从 rank 0 广播模型状态到所有其他 rank
    for param in model.parameters():
        dist.broadcast(param.data.clone(), src=0)  # 使用 clone() 方法
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model total parameters: {total_params}")
    print(f"Model trainable parameters: {trainable_params}")
    return model

def collate_fn(batch):
    text_token_ids = [item[0] for item in batch]
    audio_token_ids = [item[1] for item in batch]
    max_text_len = max(len(text) for text in text_token_ids)
    max_audio_len = max(len(audio) for audio in audio_token_ids) + 1
    chunck_length = (((max_text_len+max_audio_len) // 16) + 1) * 16
    
    text_input_ids, text_attention_mask, audio_input_ids, target, loss_mask = [], [], [], [], []
    text_pad_token = 38
    audio_pad_token = 8192
    target_pad_token = 8192
    
    for text, audio in zip(text_token_ids, audio_token_ids):
        padded_text = text + [text_pad_token] * (max_text_len - len(text))
        attention_mask = [1] * len(text) + [0] * (max_text_len - len(text))
        padded_audio = audio + [audio_pad_token] * (max_audio_len - len(audio) - 1) + [audio_pad_token] * (chunck_length - max_text_len - max_audio_len + 1)
        target_seq = [0] * (len(text) - 1) + audio + [target_pad_token] * (max_audio_len + max_text_len - len(text) - len(audio)) + [target_pad_token] * (chunck_length - max_text_len - max_audio_len + 1)
        loss_mask_seq = [0] * (len(text) - 1) + [1] * (len(audio) + 1) + [0] * (max_audio_len + max_text_len - len(text) - len(audio) - 1) + [0] * (chunck_length - max_text_len - max_audio_len+1)
        text_input_ids.append(padded_text)
        text_attention_mask.append(attention_mask)
        audio_input_ids.append(padded_audio)
        target.append(target_seq)
        loss_mask.append(loss_mask_seq)
        
    return torch.tensor(text_input_ids), torch.tensor(text_attention_mask), torch.tensor(audio_input_ids), torch.tensor(target), torch.tensor(loss_mask)

def prepare_dataloader(batch_size, rank, world_size):
    # dataset = load_dataset("JerryAGENDD/JLSpeech_tokenized", cache_dir="../temp_datasets")['train']
    # dataset = dataset.remove_columns(['text', 'audio'])
    # dataset = dataset.rename_column("normalized_text", "text_normalized")
    
    dataset2 = load_dataset("JerryAGENDD/libritts_tokenized_960", cache_dir="../temp_datasets")['train']
    dataset = dataset2.remove_columns(['text_original', 'speaker_id'])
    
    # dataset = concatenate_datasets([dataset, dataset2]).shuffle()

    dataset = MyDataset(hf_dataset=dataset, train_type='sft')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    return dataloader

def train(rank, world_size, args):
    setup(rank, world_size)
    model = initialize_model(args.checkpoint_dir, args.dim, args.n_blocks, rank,world_size)
    dataloader = prepare_dataloader(args.batch_size, rank, world_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    
    if rank == 0:
        wandb.init(project="TTS")
    logging_parameter = True
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model.train()
    for epoch in range(args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        if rank == 0:
            epoch_iterator = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=False)
        else:
            epoch_iterator = dataloader
        
        for batch in epoch_iterator:
            text_input_ids, text_attention_mask, audio_input_ids, targets, loss_masks = batch
            text_input_ids = text_input_ids.long().to(rank)
            text_attention_mask = text_attention_mask.to(rank)
            audio_input_ids = audio_input_ids.long().to(rank)
            targets = targets.long().to(rank)
            loss_masks = loss_masks.to(rank)
            outputs = model(text_input_ids, text_attention_mask, audio_input_ids)
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss = loss.view(targets.size()) * loss_masks
            loss = loss.sum() / loss_masks.sum()
            loss = L2Wrap.apply(loss, outputs)
            
            if rank == 0:
                wandb.log({"loss": loss.item()})
            
            optimizer.zero_grad()
            loss.backward()
            
            if(logging_parameter and rank == 0):
                for name, param in model.named_parameters():
                    if param.grad is None:
                        print(f"Parameter {name} did not receive gradient")

                logging_parameter = False
            
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