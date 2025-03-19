import torch
from torch.utils.data import Dataset, DataLoader
from src.text_tokenizer import TextTokenizer

class MyDataset(Dataset):
    def __init__(self, hf_dataset, train_type):
        self.hf_dataset = hf_dataset
        self.text_tokenizer = TextTokenizer()
        assert train_type in ['pretrain', 'sft']
        self.train_type = train_type
        
        
    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        data = self.hf_dataset[idx]
            
        audio_token_list = data['audio_token']
        text = data['normalized_text']
                
        if(self.train_type == 'sft'):
            text_token_list = self.text_tokenizer.tokenize(text)
            return text_token_list, audio_token_list
        else:
            
            return audio_token_list