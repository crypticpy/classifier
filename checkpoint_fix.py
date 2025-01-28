#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Checkpoint Prefix Fixer
======================

This script fixes PyTorch model checkpoints by removing '_orig_mod.' prefixes that are sometimes
added during model compilation or when using certain PyTorch features like `torch.compile()`.

Purpose:
--------
When using PyTorch's compilation features, model state dictionaries may have their keys
prefixed with '_orig_mod.', making them incompatible with the original model architecture.
This script cleanly removes these prefixes while preserving the model weights.

Usage:
------
1. Set INPUT_CHECKPOINT to your source checkpoint path
2. Set OUTPUT_CHECKPOINT to your desired output path
3. Run the script: python checkpoint_fix.py

The script handles both:
- Raw state dictionaries
- Full checkpoints containing 'model_state_dict' and other metadata

Example:
--------
Input state_dict key:  '_orig_mod.bert.embeddings.word_embeddings.weight'
Output state_dict key: 'bert.embeddings.word_embeddings.weight'

"""
import torch

INPUT_CHECKPOINT = "best_deberta_v3_weights.pt"       # The compiled checkpoint with _orig_mod. prefix
OUTPUT_CHECKPOINT = "best_deberta_v3_weights_fixed.pt" # Output after removing prefixes

def remove_orig_mod_prefix(old_dict):
    """
    old_dict might be:
      - a raw state dict
      - or a 'full checkpoint' with 'model_state_dict' inside
    Return the corrected state dict with _orig_mod. removed.
    """
    # If it's a full checkpoint with keys like 'epoch', 'model_state_dict', etc.
    if "model_state_dict" in old_dict:
        state_dict = old_dict["model_state_dict"]
    else:
        state_dict = old_dict

    new_state_dict = {}
    for key, val in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key.replace("_orig_mod.", "")
        else:
            new_key = key
        new_state_dict[new_key] = val
    return new_state_dict


def main():
    ckpt = torch.load(INPUT_CHECKPOINT, map_location="cpu")
    fixed_state_dict = remove_orig_mod_prefix(ckpt)
    torch.save(fixed_state_dict, OUTPUT_CHECKPOINT)
    print(f"Fixed checkpoint saved to: {OUTPUT_CHECKPOINT}")


if __name__ == "__main__":
    main()
