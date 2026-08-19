import tiktoken
import torch

from model.gpt import GPTModel
from model.config import GPT_CONFIG_124M
from training.generation import generate_text_simple

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

model.eval()
out = generate_text_simple(
    model = model,
    idx = encoded_tensor,
    max_new_tokens = 6, 
    context_size = GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)