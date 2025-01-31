#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
assemble_dual_model_checkpoints.py

PURPOSE:
  Creates two final model checkpoints (main + rare) for dual-model prediction.
  Each final checkpoint includes:
    - model_state_dict
    - label_encoders_<main/rare>

HOW IT WORKS:
  1) Loads the main training-data pickle (e.g., training_data.pkl) to fetch
     label encoders used for the main model.
  2) Loads the main model checkpoint (e.g., best_deberta_v3_weights.pt).
  3) Embeds "label_encoders_main" into that checkpoint dictionary.
  4) Saves it to (e.g.) best_deberta_v3_weights_main_with_encoders.pt.

  5) Repeats the process for the rare model:
     - Loads the rare-classes training-data pickle (rare_classes_data.pkl).
     - Loads best_deberta_v3_weights_rare.pt.
     - Embeds "label_encoders_rare".
     - Saves out best_deberta_v3_weights_rare_with_encoders.pt.

ARGUMENTS (via argparse):
  --main_data_pickle <path>      : Path to the main training_data.pkl
  --main_checkpoint <path>       : Path to best_deberta_v3_weights.pt
  --main_output <path>           : Output for the combined main checkpoint
  --rare_data_pickle <path>      : Path to the rare_classes_data.pkl
  --rare_checkpoint <path>       : Path to best_deberta_v3_weights_rare.pt
  --rare_output <path>           : Output for the combined rare checkpoint

EXAMPLE USAGE:
  python assemble_dual_model_checkpoints.py \
    --main_data_pickle       training_data.pkl \
    --main_checkpoint        best_deberta_v3_weights.pt \
    --main_output           best_deberta_v3_weights_main_with_encoders.pt \
    --rare_data_pickle       rare_classes_data.pkl \
    --rare_checkpoint        best_deberta_v3_weights_rare.pt \
    --rare_output           best_deberta_v3_weights_rare_with_encoders.pt

Afterward, you can load each .pt like:
  checkpoint_main = torch.load("best_deberta_v3_weights_main_with_encoders.pt")
  model_main.load_state_dict(checkpoint_main["model_state_dict"])
  label_encoders_main = checkpoint_main["label_encoders_main"]

  checkpoint_rare = torch.load("best_deberta_v3_weights_rare_with_encoders.pt")
  model_rare.load_state_dict(checkpoint_rare["model_state_dict"])
  label_encoders_rare = checkpoint_rare["label_encoders_rare"]

Author: Your Team
"""

import argparse
import pickle
import torch

def embed_encoders_in_checkpoint(
    data_pickle_path: str,
    checkpoint_path: str,
    output_path: str,
    encoder_key: str
):
    """
    Loads label encoders from `data_pickle_path` and merges them
    into `checkpoint_path`, saving out `output_path`.

    :param data_pickle_path: Path to training_data.pkl or rare_classes_data.pkl
    :param checkpoint_path: Path to best_deberta_v3_weights(.pt)
    :param output_path: Output path for combined checkpoint
    :param encoder_key: Key to store the encoders under, e.g. "label_encoders_main" or "label_encoders_rare"
    :return: None (writes out a .pt file)
    """
    print(f"Loading label encoders from: {data_pickle_path}")
    with open(data_pickle_path, "rb") as f:
        data_dict = pickle.load(f)

    if "label_encoders" not in data_dict:
        raise KeyError(f"'label_encoders' missing in {data_pickle_path}. Check your pickle structure.")

    label_encoders = data_dict["label_encoders"]

    # Summarize the found encoders
    print(f"Found label encoders in {data_pickle_path} with keys:")
    for k in label_encoders.keys():
        print(f"  - {k} -> {len(label_encoders[k].classes_)} classes")

    # Load the existing checkpoint
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # If it's raw state dict, wrap it. If it has "model_state_dict", just embed
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
        print(f"Found 'model_state_dict' in {checkpoint_path}. Embedding label encoders as '{encoder_key}'.")
        checkpoint[encoder_key] = label_encoders
    else:
        # If there's no "model_state_dict", assume it's raw weights
        model_state = checkpoint  # The entire dict is presumably the state
        print(f"No 'model_state_dict' found in {checkpoint_path}. Treating entire file as raw weights.")
        checkpoint = {
            "model_state_dict": model_state,
            encoder_key: label_encoders
        }

    # Save the new combined file
    print(f"Saving combined checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print("Done.\n")


def main():
    parser = argparse.ArgumentParser(description="Combine label encoders with main & rare model checkpoints.")
    parser.add_argument("--main_data_pickle", type=str, required=True,
                        help="Path to the main training_data.pkl.")
    parser.add_argument("--main_checkpoint", type=str, required=True,
                        help="Path to the best_deberta_v3_weights.pt for main model.")
    parser.add_argument("--main_output", type=str, default="best_deberta_v3_weights_main_with_encoders.pt",
                        help="Output path for the new combined main checkpoint.")
    parser.add_argument("--rare_data_pickle", type=str, required=True,
                        help="Path to the rare_classes_data.pkl.")
    parser.add_argument("--rare_checkpoint", type=str, required=True,
                        help="Path to the best_deberta_v3_weights_rare.pt.")
    parser.add_argument("--rare_output", type=str, default="best_deberta_v3_weights_rare_with_encoders.pt",
                        help="Output path for the new combined rare checkpoint.")

    args = parser.parse_args()

    # 1) MAIN MODEL
    print("=== EMBEDDING MAIN LABEL ENCODERS ===")
    embed_encoders_in_checkpoint(
        data_pickle_path=args.main_data_pickle,
        checkpoint_path=args.main_checkpoint,
        output_path=args.main_output,
        encoder_key="label_encoders_main"
    )

    # 2) RARE MODEL
    print("=== EMBEDDING RARE LABEL ENCODERS ===")
    embed_encoders_in_checkpoint(
        data_pickle_path=args.rare_data_pickle,
        checkpoint_path=args.rare_checkpoint,
        output_path=args.rare_output,
        encoder_key="label_encoders_rare"
    )

    print("\nAll done! You now have two final checkpoint files:")
    print(f"  1) {args.main_output}")
    print(f"  2) {args.rare_output}")
    print("Each includes both the model weights and the label encoders.\n")


if __name__ == "__main__":
    main()
