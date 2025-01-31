#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Head DeBERTa Trainer Script for Multi-Task Classification:
- Category Classification
- Subcategory Classification
- Assignment Group Classification

This script demonstrates a production-ready approach to multi-task learning
using DeBERTa V3 as a shared encoder and separate classification heads for
each task. Key aspects include:
1. Data loading and encoding with a custom Dataset.
2. Weighted sampling to address class imbalance (optional).
3. Mixed-precision training (torch.amp) for faster performance on GPUs.
4. A linear learning rate schedule with warmup for more stable training.
5. Early stopping based on validation loss to reduce overfitting.
6. Weights & Biases (wandb) integration for experiment tracking.

Why These Methods Are Good:
- **WeightedRandomSampler** helps address class imbalance by drawing
  samples in proportion to their weights, thus preventing over-representation
  of majority classes.
- **Cross Entropy Loss** is the standard for multi-class classification tasks.
- **Mixed-Precision Training** (torch.amp) allows you to speed up training
  while keeping a close approximation of full float32 accuracy.
- **Gradient Clipping** helps avoid exploding gradients, stabilizing training.
- **Early Stopping** is a simple yet effective strategy to prevent overfitting
  when validation loss stops improving.
- **Linear Warmup Schedule** gradually increases the learning rate at the start
  of training, reducing the risk of destabilizing updates.
- **AutoModel** from HuggingFace ensures you leverage proven pre-trained
  architectures, saving time and boosting performance.

Usage:
    python multi_head_deberta_trainer.py
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

# Import from torch.amp instead of torch.cuda.amp to avoid future warnings
from torch.amp import GradScaler, autocast

from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import wandb

##############################################################################
# CONFIGURATION
##############################################################################

# Path to the pickle file containing the training data
PICKLE_FILE = "training_data.pkl"

# Pre-trained DeBERTa model name from Hugging Face Transformers
MODEL_NAME = "microsoft/deberta-v3-base"

# Maximum sequence length for tokenizer
MAX_LENGTH = 512

# Number of samples per batch during training
BATCH_SIZE = 128

# Learning rate for the optimizer
LEARNING_RATE = 2e-5

# Total number of training epochs
NUM_EPOCHS = 10

# Ratio of warmup steps to total training steps
WARMUP_RATIO = 0.1

# Weight decay for the optimizer
WEIGHT_DECAY = 0.01

# Maximum gradient norm for gradient clipping
MAX_GRAD_NORM = 1.0

# Number of epochs to wait for improvement before early stopping
EARLY_STOP_PATIENCE = 3

# Flag to determine whether to use a weighted sampler (helps with class imbalance)
USE_WEIGHTED_SAMPLER = True

# Enable TensorFloat-32 (TF32) for faster computations on compatible GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

##############################################################################
# DATASET
##############################################################################

class ServiceNowDataset(Dataset):
    """
    Custom Dataset for ServiceNow data, designed for three classification tasks:
    category, subcategory, and assignment group.

    This dataset:
      - Stores the raw texts.
      - Stores three separate label lists (category, subcategory, assignment group).
      - Applies tokenization and transforms each sample into a set of tensors.

    Attributes:
        texts (List[str]): List of text samples.
        cat_labels (List[int]): List of category labels.
        subcat_labels (List[int]): List of subcategory labels.
        assign_labels (List[int]): List of assignment group labels.
        weights (List[float]): List of sample weights for addressing class imbalance.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text (DeBERTa in this case).
        max_len (int): Maximum sequence length for tokenization.
    """

    def __init__(self, texts, cat_labels, subcat_labels, assign_labels, weights, tokenizer, max_len=512):
        """
        Initializes the dataset with texts and corresponding labels.

        Args:
            texts (List[str]): List of text samples.
            cat_labels (List[int]): List of category labels (encoded as integers).
            subcat_labels (List[int]): List of subcategory labels (encoded as integers).
            assign_labels (List[int]): List of assignment group labels (encoded as integers).
            weights (List[float]): List of sample weights. These are used to mitigate
                class imbalance or place higher emphasis on certain samples.
            tokenizer (PreTrainedTokenizer): Tokenizer (here, DeBERTa) for processing text.
            max_len (int, optional): Maximum sequence length for tokenization. Defaults to 512.
        """
        self.texts = texts
        self.cat_labels = cat_labels
        self.subcat_labels = subcat_labels
        self.assign_labels = assign_labels
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves the sample at the specified index, applies tokenization,
        and returns a dictionary with all tensors needed for training.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'input_ids': Tensor of token IDs.
                - 'attention_mask': Tensor of attention mask flags (1 for real tokens).
                - 'cat_labels': Tensor of the category label.
                - 'subcat_labels': Tensor of the subcategory label.
                - 'assign_labels': Tensor of the assignment group label.
                - 'weight': Tensor containing the sample weight.
        """
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'cat_labels': torch.tensor(self.cat_labels[idx], dtype=torch.long),
            'subcat_labels': torch.tensor(self.subcat_labels[idx], dtype=torch.long),
            'assign_labels': torch.tensor(self.assign_labels[idx], dtype=torch.long),
            'weight': torch.tensor(self.weights[idx], dtype=torch.float)
        }

##############################################################################
# MODEL
##############################################################################

class DebertaV3MultiTask(nn.Module):
    """
    DeBERTa V3-based multi-task classification model with three output heads:
      1. Category classification
      2. Subcategory classification
      3. Assignment group classification

    The encoder (DeBERTa) is shared, but each task has its own linear head.
    This setup allows the backbone to learn general language representations
    while the heads specialize for each prediction task.

    Attributes:
        encoder (AutoModel): Pre-trained DeBERTa encoder.
        category_classifier (nn.Linear): Linear layer for category classification.
        subcategory_classifier (nn.Linear): Linear layer for subcategory classification.
        assignment_group_classifier (nn.Linear): Linear layer for assignment group classification.
    """

    def __init__(self, model_name, num_categories, num_subcategories, num_assignment_groups):
        """
        Initializes the multi-task model.

        Args:
            model_name (str): Name of the pre-trained DeBERTa model (from HuggingFace).
            num_categories (int): Number of category classes.
            num_subcategories (int): Number of subcategory classes.
            num_assignment_groups (int): Number of assignment group classes.
        """
        super(DebertaV3MultiTask, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # Typically 768 for DeBERTa-base

        # Classification heads for each task
        self.category_classifier = nn.Linear(hidden_size, num_categories)
        self.subcategory_classifier = nn.Linear(hidden_size, num_subcategories)
        self.assignment_group_classifier = nn.Linear(hidden_size, num_assignment_groups)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs (batch_size x sequence_length).
            attention_mask (torch.Tensor): Tensor of attention masks (batch_size x sequence_length).

        Returns:
            tuple: (cat_logits, subcat_logits, assign_logits)
                - cat_logits: Category logits (batch_size x num_categories)
                - subcat_logits: Subcategory logits (batch_size x num_subcategories)
                - assign_logits: Assignment group logits (batch_size x num_assignment_groups)
        """
        # Encoder outputs: (last_hidden_state, pooler_output, hidden_states)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the first token ([CLS]) representation from the final layer
        # This is often used as a summary of the entire sequence
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Compute classification logits for each task
        cat_logits = self.category_classifier(pooled_output)
        subcat_logits = self.subcategory_classifier(pooled_output)
        assign_logits = self.assignment_group_classifier(pooled_output)

        return cat_logits, subcat_logits, assign_logits

##############################################################################
# LOSS & EVALUATION
##############################################################################

def compute_loss_and_outputs(batch, model, device):
    """
    Computes the loss and model outputs for a given batch.
    This function centralizes the forward pass and cross-entropy computations.

    Why Weighted Loss?
      - By applying 'reduction="none"', we can multiply each sample's loss
        by its corresponding weight, allowing the model to pay more
        attention to rare or important samples.

    Args:
        batch (dict): Dictionary containing input tensors and labels from the dataset.
        model (nn.Module): The multi-task classification model.
        device (torch.device): Device (CPU or CUDA) to perform computations on.

    Returns:
        tuple:
            - total_loss (torch.Tensor): Combined mean loss from the three tasks.
            - (cat_logits, subcat_logits, assign_logits): Logits from each task head.
    """
    # Move inputs and labels to the specified device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    cat_labels = batch['cat_labels'].to(device)
    subcat_labels = batch['subcat_labels'].to(device)
    assign_labels = batch['assign_labels'].to(device)
    sample_weights = batch['weight'].to(device)

    # Forward pass to get logits from each classification head
    cat_logits, subcat_logits, assign_logits = model(input_ids, attention_mask)

    # Compute individual cross-entropy losses with 'none' reduction
    cat_loss = F.cross_entropy(cat_logits, cat_labels, reduction='none')
    subcat_loss = F.cross_entropy(subcat_logits, subcat_labels, reduction='none')
    assign_loss = F.cross_entropy(assign_logits, assign_labels, reduction='none')

    # Apply sample-specific weights to each loss term
    cat_loss = (cat_loss * sample_weights).mean()
    subcat_loss = (subcat_loss * sample_weights).mean()
    assign_loss = (assign_loss * sample_weights).mean()

    # Combine all task losses equally
    total_loss = (cat_loss + subcat_loss + assign_loss) / 3.0

    return total_loss, (cat_logits, subcat_logits, assign_logits)

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluates the model on a validation dataset to compute average loss
    and accuracy metrics for all three tasks.

    Args:
        model (nn.Module): The multi-task classification model.
        dataloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform computations on.

    Returns:
        tuple: (avg_loss, cat_acc, subcat_acc, assign_acc)
            - avg_loss (float): Average loss across the dataset.
            - cat_acc (float): Classification accuracy for category.
            - subcat_acc (float): Classification accuracy for subcategory.
            - assign_acc (float): Classification accuracy for assignment group.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_samples = 0

    cat_correct = 0
    subcat_correct = 0
    assign_correct = 0

    # Iterate over the validation batches
    for batch in dataloader:
        # Mixed-precision context for efficient inference
        with autocast(device_type='cuda'):
            loss, (cat_logits, subcat_logits, assign_logits) = compute_loss_and_outputs(batch, model, device)

        total_loss += loss.item()

        # Retrieve labels
        cat_labels = batch['cat_labels'].to(device)
        subcat_labels = batch['subcat_labels'].to(device)
        assign_labels = batch['assign_labels'].to(device)

        # Predictions: argmax over logits for each task
        cat_preds = torch.argmax(cat_logits, dim=1)
        subcat_preds = torch.argmax(subcat_logits, dim=1)
        assign_preds = torch.argmax(assign_logits, dim=1)

        # Count how many predictions match the labels
        cat_correct += (cat_preds == cat_labels).sum().item()
        subcat_correct += (subcat_preds == subcat_labels).sum().item()
        assign_correct += (assign_preds == assign_labels).sum().item()

        total_samples += cat_labels.size(0)

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    cat_acc = cat_correct / total_samples
    subcat_acc = subcat_correct / total_samples
    assign_acc = assign_correct / total_samples

    return avg_loss, cat_acc, subcat_acc, assign_acc

##############################################################################
# MAIN TRAINING LOOP
##############################################################################

def main():
    """
    Main function to orchestrate data loading, model training, evaluation, and logging.

    Steps:
      1. Determine device (GPU/CPU).
      2. Initialize Weights & Biases for experiment tracking (optional).
      3. Load data from pickle and split into train/val sets.
      4. Create custom datasets and corresponding data loaders.
      5. Initialize the multi-task model and optimizer.
      6. Train the model with mixed-precision, gradient clipping, and scheduler.
      7. Perform validation after each epoch, with early stopping.
      8. Save and load the best model weights for final usage.

    This approach is robust for multi-task classification and scales well
    for moderate to large datasets due to the memory efficiencies of
    mixed-precision training and TF32 acceleration (if available).
    """
    # Determine the device to run on (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Initialize Weights & Biases (wandb) for experiment tracking
    try:
        # Provide your wandb API key if you want to track experiments
        wandb.login(key="YOUR_KEY_HERE")
        wandb.init(project="deberta_multitask", name="Multitask_3Heads_Train")
    except Exception as e:
        print(f"wandb initialization failed: {e}")
        pass

    print("Loading data from pickle...")
    # Load training data from the specified pickle file
    with open(PICKLE_FILE, 'rb') as f:
        data_dict = pickle.load(f)

    # Extract data components from the loaded dictionary
    texts = data_dict['text']
    cat_labels = data_dict['category_labels']
    subcat_labels = data_dict['subcategory_labels']
    assign_labels = data_dict['assignment_group_labels']
    sample_weights = data_dict['sample_weights']

    # Determine the number of classes for each task (using label encoders)
    num_categories = len(data_dict['label_encoders']['Category'].classes_)
    num_subcategories = len(data_dict['label_encoders']['Subcategory'].classes_)
    num_assignment_groups = len(data_dict['label_encoders']['Assignment Group'].classes_)

    # Print dataset statistics
    print(f"Samples: {len(texts)}")
    print(f"Categories: {num_categories}, Subcategories: {num_subcategories}, AssignmentGroups: {num_assignment_groups}")

    # Split data into training and validation sets (10% validation)
    # Using stratify on category labels to maintain distribution in each split
    indices = list(range(len(texts)))
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, stratify=cat_labels)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Create training dataset
    train_dataset = ServiceNowDataset(
        [texts[i] for i in train_idx],
        [cat_labels[i] for i in train_idx],
        [subcat_labels[i] for i in train_idx],
        [assign_labels[i] for i in train_idx],
        [sample_weights[i] for i in train_idx],
        tokenizer,
        MAX_LENGTH
    )

    # Create validation dataset
    val_dataset = ServiceNowDataset(
        [texts[i] for i in val_idx],
        [cat_labels[i] for i in val_idx],
        [subcat_labels[i] for i in val_idx],
        [assign_labels[i] for i in val_idx],
        [sample_weights[i] for i in val_idx],
        tokenizer,
        MAX_LENGTH
    )

    # Initialize DataLoader for training
    # WeightedRandomSampler can help if there's a large imbalance in label distribution
    if USE_WEIGHTED_SAMPLER:
        train_weights = [sample_weights[i] for i in train_idx]
        sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=8,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=RandomSampler(train_dataset),
            num_workers=8,
            pin_memory=True
        )

    # Initialize DataLoader for validation
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=SequentialSampler(val_dataset),
        num_workers=8,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Initialize the multi-task classification model
    model = DebertaV3MultiTask(
        model_name=MODEL_NAME,
        num_categories=num_categories,
        num_subcategories=num_subcategories,
        num_assignment_groups=num_assignment_groups
    ).to(device)

    # Optionally compile the model with PyTorch 2.0 to potentially gain speed improvements
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # Initialize the optimizer (AdamW) with learning rate and weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Calculate total training steps and warmup steps
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    # Set up a linear schedule with a warmup period
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Initialize a gradient scaler for mixed-precision training
    scaler = GradScaler()

    # Variables to track the best validation loss and early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = "best_deberta_v3_weights.pt"

    global_step = 0
    print("Starting training...")
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()  # Set model to training mode
        total_train_loss = 0.0

        # Initialize progress bar for the epoch
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for step, batch in enumerate(epoch_iterator):
            # Automatic Mixed Precision context
            with autocast(device_type='cuda'):
                loss, _ = compute_loss_and_outputs(batch, model, device)

            # Scale the loss for backprop
            scaler.scale(loss).backward()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()

            # Step the learning rate scheduler
            scheduler.step()

            # Zero out the gradients for the next iteration
            optimizer.zero_grad()

            # Accumulate training loss
            total_train_loss += loss.item()
            global_step += 1

            # Update progress bar
            epoch_iterator.set_postfix(loss=f"{(total_train_loss/(step+1)):.4f}")

            # Occasionally clean up GPU memory
            if step % 200 == 0:
                torch.cuda.empty_cache()

        # Calculate average training loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluate on validation set
        val_loss, cat_acc, subcat_acc, assign_acc = evaluate(model, val_loader, device)

        # Log metrics to Weights & Biases (if running)
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "cat_acc": cat_acc,
            "subcat_acc": subcat_acc,
            "assign_acc": assign_acc
        }, step=epoch+1)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Val Acc: Cat={cat_acc:.4f}, Subcat={subcat_acc:.4f}, Assign={assign_acc:.4f}")

        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset the patience counter
            torch.save(model.state_dict(), best_model_path)  # Save the best model
            print(f"  [Saved best model @ val_loss={val_loss:.4f}]")
        else:
            patience_counter += 1
            print(f"  [No improvement. Patience={patience_counter}/{EARLY_STOP_PATIENCE}]")
            # Early stopping if we exceed the patience threshold
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training complete.")

    # Load the best model weights for final usage or further evaluation
    model.load_state_dict(torch.load(best_model_path))
    print("Loaded best model weights for final usage.")

if __name__ == "__main__":
    main()
