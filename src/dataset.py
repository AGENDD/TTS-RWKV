import torch
from torch.utils.data import Dataset, DataLoader
from src.text_tokenizer import TextTokenizer

class MyDataset(Dataset):
    """
    A PyTorch Dataset for handling tokenized text and audio data.
    This dataset supports two training modes:
    - 'pretrain': Returns only audio tokens for self-supervised pre-training
    - 'sft': Returns both text and audio tokens for supervised fine-tuning
    Args:
        hf_dataset: A Hugging Face dataset containing 'audio_token' and 'normalized_text' fields
        train_type (str): The training type, either 'pretrain' or 'sft'
    Returns:
        For 'pretrain' mode:
            - audio_token_list: List of audio tokens
        For 'sft' mode:
            - text_token_list: List of tokenized text
            - audio_token_list: List of audio tokens
    Note:
        - Requires a TextTokenizer class to tokenize text data
        - The 'audio_token' and 'normalized_text' fields must be present in the dataset
    """
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