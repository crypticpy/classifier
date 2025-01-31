#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
rare_classes_preparation.py

Purpose:
--------
Produce a dataset of only the "rare" classes—those that would otherwise be merged
into "Other"—and heavily augment them. We then label-encode and compute sample
weights, outputting a pickle with the same structure as your main data-preparation.

Key Enhancements:
----------------
- If a class is "heavy" (size < HEAVY_AUG_THRESHOLD), we multiply the baseline
  RARE_MULTIPLIER by an additional HEAVY_BONUS_MULTIPLIER to create extra passes.
- Each sample's augmentation is randomized, so repeated passes produce unique
  variations.
- Fast augmentations (synonym, insertion, deletion, swap) are performed first
- Backtranslation is performed in parallel using MarianMT models directly

Usage:
------
python rare_classes_preparation.py
"""

import re
import random
import pickle
import pandas as pd
import numpy as np
import nltk
import torch
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import nlpaug.augmenter.word as naw
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from transformers import MarianMTModel, MarianTokenizer

# Download NLTK resources if needed
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

##############################################################################
# CONFIGURATION
##############################################################################
EXCEL_FILE = "data_18_mos.xlsx"
OUTPUT_RARE_PKL = "rare_classes_data.pkl"
# Thresholds for identifying "rare" classes
MIN_SAMPLES_CATEGORY = 50
MIN_SAMPLES_SUBCATEGORY = 50
MIN_SAMPLES_ASSIGNMENT = 50

# Text augmentation multipliers/params
RARE_MULTIPLIER = 5         # how many augmented samples to create per row
HEAVY_AUG_THRESHOLD = 25    # classes smaller than this get "heavy" augmentation
HEAVY_BONUS_MULTIPLIER = 2  # extra factor for heavy classes

# Parallel processing settings
NUM_PROCESSES = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
BATCH_SIZE = 16  # Size of batches for translation

# Max sample weight
MAX_WEIGHT = 10.0

##############################################################################
# DATA LOADING & BASIC CLEANING
##############################################################################
def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_excel(filepath)

def basic_clean_text(text):
    """
    Remove emails, URLs, special characters;
    standardize numbers/whitespace; lowercase.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    text = re.sub(r'http\S+|www.\S+', 'URL', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '[NUM]', text)
    text = ' '.join(text.split())
    return text.strip().lower()

##############################################################################
# IDENTIFY RARE CLASSES
##############################################################################
def get_rare_classes(df: pd.DataFrame,
                     col: str,
                     min_samples: int) -> set:
    """
    Return a set of class labels that have fewer than `min_samples` in `col`.
    """
    counts = df[col].value_counts()
    return set(counts[counts < min_samples].index)

##############################################################################
# TEXT AUGMENTATION STRATEGIES
##############################################################################
def get_synonyms(word: str) -> list:
    """
    Retrieve synonyms for a word from WordNet.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            if candidate != word:
                synonyms.add(candidate)
    return list(synonyms)

def random_synonym_replace(text: str, ratio=0.15) -> str:
    """
    Replace a fraction of words with synonyms.
    """
    words = text.split()
    if not words:
        return text
    n_replace = max(1, int(len(words) * ratio))
    idx_candidates = [i for i, w in enumerate(words) if w.isalpha() and len(w) > 3]
    if not idx_candidates:
        return text
    
    to_replace = random.sample(idx_candidates, min(n_replace, len(idx_candidates)))
    for idx in to_replace:
        syns = get_synonyms(words[idx])
        if syns:
            words[idx] = random.choice(syns)
    return " ".join(words)

def random_insertion(text: str, n=1) -> str:
    words = text.split()
    if not words:
        return text
    for _ in range(n):
        idx = random.randint(0, len(words) - 1)
        syns = get_synonyms(words[idx])
        if syns:
            insert_word = random.choice(syns)
            insert_pos = random.randint(0, len(words))
            words.insert(insert_pos, insert_word)
    return " ".join(words)

def random_deletion(text: str, p=0.1) -> str:
    words = text.split()
    if len(words) < 4:
        return text
    kept = [w for w in words if random.random() > p]
    if not kept:
        kept = random.sample(words, 1)
    return " ".join(kept)

def random_swap_phrases(text: str, segment_count=3) -> str:
    words = text.split()
    if len(words) < segment_count:
        return text
    seg_size = len(words) // segment_count
    segments = [words[i : i + seg_size] for i in range(0, len(words), seg_size)]
    random.shuffle(segments)
    return " ".join(w for seg in segments for w in seg)

def fast_augment(text: str, heavy: bool = False) -> str:
    """
    Combine multiple fast augmentation steps
    """
    # 1) Synonym replacement
    ratio = 0.20 if heavy else 0.10
    text = random_synonym_replace(text, ratio=ratio)
    
    # 2) Insert or delete
    if heavy:
        text = random_insertion(text, n=1)
    else:
        text = random_deletion(text, p=0.1)
    
    # 3) 50% chance to shuffle
    if random.random() < 0.5:
        text = random_swap_phrases(text, segment_count=3)
    
    return text

def translate_batch(texts: list, model: MarianMTModel, tokenizer: MarianTokenizer, device: str) -> list:
    """Translate a batch of texts"""
    # Tokenize
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, 
                      max_length=512).to(device)
    
    # Generate translation
    translated = model.generate(**inputs)
    
    # Decode
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_texts

def parallel_backtranslate(texts: list) -> list:
    """
    Perform backtranslation using nlpaug
    """
    print("Loading translation models...")
    aug = naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-de',
        to_model_name='Helsinki-NLP/opus-mt-de-en',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=BATCH_SIZE,
        max_length=512
    )
    
    # Process all texts at once with tqdm for progress
    print(f"Translating {len(texts)} texts...")
    translated_texts = []
    
    # Process in smaller chunks to show progress
    chunk_size = 100
    for i in tqdm(range(0, len(texts), chunk_size), desc="Backtranslation"):
        chunk = texts[i:i + chunk_size]
        augmented = aug.augment(chunk)
        translated_texts.extend(augmented[0] if isinstance(augmented[0], list) else augmented)
    
    return translated_texts

##############################################################################
# AUGMENTATION FOR RARE CLASSES
##############################################################################
def augment_rare_rows(df: pd.DataFrame,
                      rare_cat: set,
                      rare_subcat: set,
                      rare_assign: set) -> pd.DataFrame:
    """
    Two-phase augmentation:
    1. Fast augmentations (synonym, insert, delete, swap)
    2. Parallel backtranslation on a subset of samples
    """
    # Pre-check class sizes
    cat_counts = df['Category'].value_counts()
    subcat_counts = df['Subcategory'].value_counts()
    assign_counts = df['Assignment Group'].value_counts()
    
    augmented_rows = []
    texts_for_backtranslation = []
    backtranslation_indices = []
    
    print("Phase 1: Performing fast augmentations...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fast Augmentations"):
        c = row['Category']
        s = row['Subcategory']
        a = row['Assignment Group']
        
        # If row is not in any rare class, skip
        if c not in rare_cat and s not in rare_subcat and a not in rare_assign:
            continue
        
        # Decide if 'heavy' augmentation is needed
        is_heavy = (
            (c in rare_cat and cat_counts[c] < HEAVY_AUG_THRESHOLD) or
            (s in rare_subcat and subcat_counts[s] < HEAVY_AUG_THRESHOLD) or
            (a in rare_assign and assign_counts[a] < HEAVY_AUG_THRESHOLD)
        )
        
        # base multiplier
        n_new = RARE_MULTIPLIER
        if is_heavy:
            n_new *= HEAVY_BONUS_MULTIPLIER
        
        for i in range(n_new):
            new_row = row.copy()
            new_row['combined_text'] = fast_augment(new_row['combined_text'], heavy=is_heavy)
            augmented_rows.append(new_row)
            
            # Randomly select 50% of augmented samples for backtranslation
            if random.random() < 0.5:
                texts_for_backtranslation.append(new_row['combined_text'])
                backtranslation_indices.append(len(augmented_rows) - 1)
    
    # Convert to DataFrame
    df_aug = pd.DataFrame(augmented_rows)
    if df_aug.empty:
        print("No rows belonged to rare classes or no augmented samples produced.")
        return df
    
    print(f"\nPhase 2: Performing parallel backtranslation on {len(texts_for_backtranslation)} samples...")
    if texts_for_backtranslation:
        # Process backtranslations in batches
        translated_texts = parallel_backtranslate(texts_for_backtranslation)
        
        # Update the augmented texts with backtranslations
        for idx, trans_text in zip(backtranslation_indices, translated_texts):
            df_aug.iloc[idx, df_aug.columns.get_loc('combined_text')] = trans_text
    
    print(f"Augmentation complete. Final shape: {df_aug.shape}")
    return pd.concat([df, df_aug], ignore_index=True)

##############################################################################
# SAMPLE WEIGHTS
##############################################################################
def compute_sample_weights(df: pd.DataFrame,
                           cat_col='Category',
                           subcat_col='Subcategory',
                           assign_col='Assignment Group',
                           max_weight=MAX_WEIGHT) -> pd.Series:
    """
    Compute sample weights as in your main script:
    - Weighted by sqrt(1/freq) for each of the 3 label columns
    - Then geometric mean
    - Clipped to [1, max_weight]
    """
    cat_counts = df[cat_col].value_counts()
    subcat_counts = df[subcat_col].value_counts()
    assign_counts = df[assign_col].value_counts()
    
    cat_weights = np.sqrt(1.0 / cat_counts)
    cat_weights = cat_weights / cat_weights.min()
    
    subcat_weights = np.sqrt(1.0 / subcat_counts)
    subcat_weights = subcat_weights / subcat_weights.min()
    
    assign_weights = np.sqrt(1.0 / assign_counts)
    assign_weights = assign_weights / assign_weights.min()
    
    def row_wt(row):
        w_cat = cat_weights.get(row[cat_col], 1.0)
        w_sub = subcat_weights.get(row[subcat_col], 1.0)
        w_asg = assign_weights.get(row[assign_col], 1.0)
        w = (w_cat * w_sub * w_asg) ** (1/3)
        return min(max_weight, max(1.0, w))
    
    return df.apply(row_wt, axis=1)

##############################################################################
# MAIN
##############################################################################
def main():
    print(f"Loading data from {EXCEL_FILE}...")
    df = load_data(EXCEL_FILE)
    print(f"Initial shape: {df.shape}")
    
    # Basic cleaning for relevant text fields
    df['Short Description'] = df['Short Description'].fillna('').apply(basic_clean_text)
    df['Description'] = df['Description'].fillna('').apply(basic_clean_text)
    
    # Drop rows missing any of the 3 target columns
    df = df.dropna(subset=['Category', 'Subcategory', 'Assignment Group'])
    print(f"After dropping missing targets: {df.shape}")
    
    # Build combined_text if not present
    if 'combined_text' not in df.columns:
        df['combined_text'] = (
            df['Short Description'] + ' [SEP] ' +
            df['Description']
        )
    
    # Identify which classes are "rare" in each target
    rare_cat = get_rare_classes(df, 'Category', MIN_SAMPLES_CATEGORY)
    rare_subcat = get_rare_classes(df, 'Subcategory', MIN_SAMPLES_SUBCATEGORY)
    rare_assign = get_rare_classes(df, 'Assignment Group', MIN_SAMPLES_ASSIGNMENT)
    
    # Filter to only rows that belong to at least one rare class
    initial_len = len(df)
    df_rare = df[
        df['Category'].isin(rare_cat) |
        df['Subcategory'].isin(rare_subcat) |
        df['Assignment Group'].isin(rare_assign)
    ]
    print(f"Rare-class subset shape: {df_rare.shape} (out of {initial_len}).")
    
    # Augment
    df_aug = augment_rare_rows(df_rare, rare_cat, rare_subcat, rare_assign)
    print(f"Shape after augmentation: {df_aug.shape}")
    
    # Compute sample weights
    df_aug['sample_weight'] = compute_sample_weights(df_aug)
    
    # Label-encode each target to preserve distinct classes
    label_encoders = {}
    for col in ['Category', 'Subcategory', 'Assignment Group']:
        le = LabelEncoder()
        df_aug[col] = le.fit_transform(df_aug[col])
        label_encoders[col] = le
    
    # Build final dictionary matching your main script
    train_dict = {
        'text': df_aug['combined_text'].tolist(),
        'category_labels': df_aug['Category'].tolist(),
        'subcategory_labels': df_aug['Subcategory'].tolist(),
        'assignment_group_labels': df_aug['Assignment Group'].tolist(),
        'sample_weights': df_aug['sample_weight'].tolist(),
        'label_encoders': label_encoders
    }
    
    # Save to pickle
    with open(OUTPUT_RARE_PKL, 'wb') as f:
        pickle.dump(train_dict, f)
    
    print(f"\nRare-class data and augmentations saved to '{OUTPUT_RARE_PKL}'.")
    print(f"Final shape: {df_aug.shape}. Number of rare classes retained:")
    print(f"Category: {len(rare_cat)}, Subcategory: {len(rare_subcat)}, "
          f"Assignment Group: {len(rare_assign)}")

def test_backtranslation():
    """Test function to verify backtranslation works"""
    test_texts = [
        "This is a test sentence for backtranslation.",
        "Another example to verify the translation works properly."
    ]
    print("\nTesting backtranslation with sample texts:")
    print("Original texts:", test_texts)
    translated = parallel_backtranslate(test_texts)
    print("Translated texts:", translated)
    return translated

if __name__ == "__main__":
    # Run test first
    test_backtranslation()
    
    # Then run main
    main()
