import torch
import tiktoken

from model.gpt import GPTModel
from model.config import GPT2_SMALL_CONFIG

from training.train import train_model_simple
from training.generation import generate_outputs

from arxiv.find_main import find_main_tex
from arxiv.preprocess import preprocess_file

from data.data_loader import create_dataloader_v1

# --------------------------------------------
# CONFIGURATION
# --------------------------------------------

NUM_EPOCHS = 3
BATCH_SIZE = 2
CONTEXT_LENGTH = GPT2_SMALL_CONFIG["context_length"]

EVAL_FREQ = 10
EVAL_ITER = 10

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.1

CHECKPOINT_PATH = "checkpoints/gpt2_research_pretrained.pth"
OUTPUT_BEFORE = "examples/output_before_pretraining.txt"
OUTPUT_AFTER = "examples/output_after_pretraining.txt"

# --------------------------------------------
# DEVICE, TOKENIZER AND MODEL
# --------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)
gpt = GPTModel(GPT2_SMALL_CONFIG)
optimizer = torch.optim.AdamW(
    gpt.parameters(), 
    lr = LEARNING_RATE, 
    weight_decay = WEIGHT_DECAY
)

state_dict = torch.load(
    "checkpoints/gpt2_124M.pth",
    map_location = device,
    weights_only = True
)

gpt.load_state_dict(state_dict)
gpt.to(device)
gpt.eval()

# --------------------------------------------
# EVALUATION BEFORE PRETRAINING
# --------------------------------------------

with open("examples/inputs.txt", "r", encoding = "utf-8") as file:
    inputs = file.read().splitlines()

generate_outputs(
    model = gpt,
    tokenizer = tokenizer,
    device = device,
    inputs = inputs,
    max_new_tokens = 25, 
    temperature = 0, 
    output_path = OUTPUT_BEFORE
)

# --------------------------------------------
# DATASET
# --------------------------------------------

arxiv_id = "2102.01111"
main_file = find_main_tex(arxiv_id)
text_data = preprocess_file(main_file)

train_ratio = 0.9
split_idx = int(train_ratio*len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data,
    batch_size = BATCH_SIZE,
    max_length = CONTEXT_LENGTH,
    stride = CONTEXT_LENGTH,
    drop_last = True,
    shuffle = True,
    num_workers = 0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size = BATCH_SIZE,
    max_length = CONTEXT_LENGTH,
    stride = CONTEXT_LENGTH,
    drop_last = False,
    shuffle = False,
    num_workers = 0
)

# --------------------------------------------
# FURTHER PRETRAINING
# --------------------------------------------

train_losses, val_losses, tokens_seen = train_model_simple(
    gpt, train_loader, val_loader, optimizer, device,
    num_epochs = NUM_EPOCHS, eval_freq = EVAL_FREQ, eval_iter = EVAL_ITER
)

# --------------------------------------------
# SAVE CHECKPOINT
# --------------------------------------------

torch.save({
    "model_state_dict": gpt.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    CHECKPOINT_PATH
)

# --------------------------------------------
# EVALUATION AFTER PRETRAINING
# --------------------------------------------

gpt.eval()

generate_outputs(
    model = gpt,
    tokenizer = tokenizer,
    device = device,
    inputs = inputs,
    max_new_tokens = 25, 
    temperature = 0, 
    output_path = OUTPUT_AFTER
)