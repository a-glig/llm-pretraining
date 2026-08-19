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