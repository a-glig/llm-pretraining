import torch.nn as nn
import matplotlib.pyplot as plt


def calc_loss_batch(input_batch, target_batch, model, device):
    """Training and validation loss for a single batch."""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches = None):
    """Training and validation loss for entire dataset"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break

    return total_loss/num_batches


def plot_losses(train_losses, val_losses, tokens_seen):
    """
    Plot training and validation loss against the number of tokens used 
    for training.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(tokens_seen, train_losses, label="Training loss")
    ax.plot(
        tokens_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax.set_xlabel("Tokens seen")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("losses.png", dpi=300, bbox_inches="tight")
    plt.show()