import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    A function for the GPT model to generate text. Appends the token that has 
    the highest probability to follow after the input. 
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            # Tensor shape: (b, n_token, vocab_size)
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim = -1)
        idx_next = torch.argmax(probas, dim = -1, keepdim = True)
        idx = torch.cat((idx, idx_next), dim = 1)

    return idx


def generate(model, idx, max_new_tokens, context_size, 
             temperature = 0.0, top_k = None, eos_id = None):
    """
    Function for generating text that includes temperature scaling and 
    top-k sampling.    
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, top_pos = torch.topk(logits, top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probas, num_samples = 1)
        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim = 1)

    return idx
    

def text_to_token_ids(text, tokenizer):
    """Helper function for conversion from text to token IDs."""
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    # GPT model expects tensor (b, num_tokens)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """Helper function for conversion from token IDs to text."""
    # Decode method expects 1d tensor
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generates sample text to track the model's progress during training.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model = model, idx = encoded, 
            max_new_tokens = 50, context_size = context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()