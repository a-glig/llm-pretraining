import json
import os
import torch
import tensorflow as tf

from gpt_download import load_gpt2_params_from_tf_ckpt
from model.gpt import GPTModel
from model.weights import load_weights_into_gpt
from model.config import GPT2_SMALL_CONFIG

# Location of original GPT-2 checkpoint
checkpoint_dir = "gpt2/124M"

# Find TensorFlow checkpoint
tf_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir)

# Load GPT-2 configuration
with open(os.path.join(checkpoint_dir, "hparams.json"), "r") as f:
    settings = json.load(f)

# Convert TensorFlow checkpoint to parameter dictionary
params = load_gpt2_params_from_tf_ckpt(
    tf_ckpt_path,
    settings
)

# Create GPT-2 model
model = GPTModel(GPT2_SMALL_CONFIG)

# Put GPT-2 weights into the model
load_weights_into_gpt(model, params)

# Save as PyTorch checkpoint
os.makedirs("checkpoints", exist_ok = True)

torch.save(
    model.state_dict(),
    "checkpoints/gpt2_124M.pth"
)

