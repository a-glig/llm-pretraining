import tiktoken
from torch.utils.data import DataLoader

from .dataset import GPTDatasetV1

def create_dataloader_v1(txt, batch_size = 4, max_length = 256,
                         stride = 128, shuffle = True, drop_last = True, 
                         num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last, 
        num_workers = num_workers
    )

    return dataloader