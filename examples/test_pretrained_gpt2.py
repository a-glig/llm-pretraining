import torch
import tiktoken

from model.gpt import GPTModel
from model.config import GPT2_SMALL_CONFIG
from training.generation import generate, text_to_token_ids, token_ids_to_text

# Initialize model and relevant components
gpt = GPTModel(GPT2_SMALL_CONFIG)
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained GPT-2 weights
state_dict = torch.load(
    "checkpoints/gpt2_124M.pth",
    map_location= device,
    weights_only = True
)
gpt.load_state_dict(state_dict)
gpt.to(device)
gpt.eval()

# Generate text
torch.manual_seed(123)
token_ids = generate(
    model = gpt,
    idx = text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens = 25,
    context_size = GPT2_SMALL_CONFIG["context_length"],
    top_k = 50,
    temperature = 1.5
)

# Decode and display output
decoded_text = token_ids_to_text(token_ids, tokenizer)
print("Output text:\n", decoded_text)