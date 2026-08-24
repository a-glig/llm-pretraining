import torch
import tiktoken

from model.gpt import GPTModel
from model.config import GPT_CONFIG_124M
from training.train import train_model_simple
from data.data_loader import create_dataloader_v1

# Initialize model and relevant components
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 0.0004, weight_decay = 0.1
)

# Create training and validation dataset (Short story "The verdict").
file_path = "examples/the-verdict.txt"
with open(file_path, "r", encoding = "utf-8") as file:
    text_data = file.read()

train_ratio = 0.9
split_idx = int(train_ratio*len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size = 2,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last = True,
    shuffle = True,
    num_workers = 0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size = 2,
    max_length = GPT_CONFIG_124M["context_length"],
    stride = GPT_CONFIG_124M["context_length"],
    drop_last = False,
    shuffle = False,
    num_workers = 0
)

# Pretrain and evaluate model 
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs = num_epochs, eval_freq = 5, eval_iter = 5,
    start_context = "Every effort moves you", tokenizer = tokenizer 
)
