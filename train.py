import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import glob
import wandb
from datasets import load_dataset
from accelerate import Accelerator

from src.rwkv7 import RWKV7
from src.dataset import MyDataset

######### build model #################
model = RWKV7(text_vocab=128, audio_vocab=8192 + 1, dim=128, n_blocks=5).cuda()
# model = torch.nn.DataParallel(model)  # 添加这一行以支持多卡训练

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

# 打印模型总体参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Model total parameters: {total_params}")

# 打印训练参数量和训练参数名字
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
trainable_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
print(f"Model train parameters: {trainable_params}")
# print(f"训练参数名字: {trainable_param_names}")

######### build dataloader #################
def collate_fn(batch):
    padding_token = 8192
    max_length = max(len(seq) for seq in batch) - 1  # 最大长度，不包括最后一个token

    input_ids = []
    targets = []
    loss_masks = []

    for seq in batch:
        input_seq = list(seq[:-1])  # 输入序列，不包括最后一个token
        target_seq = list(seq[1:])  # 目标序列，从第二个token开始
        input_padding = [padding_token] * (max_length - len(input_seq))
        target_padding = [padding_token] * (max_length - len(target_seq))
        mask_padding = [0] * (max_length - len(input_seq))

        input_ids.append(torch.tensor(input_seq + input_padding, dtype=torch.long))
        targets.append(torch.tensor(target_seq + target_padding, dtype=torch.long))
        loss_masks.append(torch.tensor([1] * len(input_seq) + mask_padding, dtype=torch.long))

    return torch.stack(input_ids, dim=0), torch.stack(targets, dim=0), torch.stack(loss_masks, dim=0)

dataset = load_dataset("JerryAGENDD/JLSpeech_tokenized", cache_dir="../temp_datasets")['train']
dataloader = MyDataset(hf_dataset=dataset, train_type='pretrain')
dataloader = DataLoader(dataloader, batch_size=128, shuffle=True, collate_fn=collate_fn)
# print(dataset)

######### configuration #################
accelerator = Accelerator()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

######### training #################
num_epochs = 4000
output_dir = "./checkpoints"
os.makedirs(output_dir, exist_ok=True)

wandb.init(project="TTS")


model.train()
for epoch in tqdm(range(num_epochs)):
    for batch in tqdm(dataloader, leave=False):
        input_ids, targets, loss_masks = batch
        input_ids = input_ids.long().to('cuda')
        targets = targets.long().to('cuda')
        loss_masks = loss_masks.to('cuda')

        # 前向传播
        outputs = model(None, input_ids)
        # 计算损失
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        # 将loss_masks应用于损失
        loss = loss.view(targets.size()) * loss_masks
        loss = loss.sum() / loss_masks.sum()  # 计算平均损失

        # tqdm.write(str(loss))
        wandb.log({"loss": loss.item()})
        # 反向传播和优化
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

    
    # 删除目标路径下所有pt文件
    pt_files = glob.glob(os.path.join(output_dir, "*.pt"))
    for pt_file in pt_files:
        os.remove(pt_file)

    # 保存当前检查点
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
    torch.save(model.state_dict(), checkpoint_path)

# Finish the wandb run
wandb.finish()