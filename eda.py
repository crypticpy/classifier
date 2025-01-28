#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import re
import nltk
import random
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources if not present
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import wordnet

##############################################################################
# CONFIGURATION
##############################################################################
EXCEL_FILE = "data_18_mos.xlsx"
OUTPUT_PKL = "training_data.pkl"

# Rare class thresholds
MIN_SAMPLES_CATEGORY = 50
MIN_SAMPLES_SUBCATEGORY = 50
MIN_SAMPLES_ASSIGNMENT = 50

# Synonym-based oversampling parameters
OVERSAMPLE_MULTIPLIER = 2       # replicate how many times
THRESHOLD_FACTOR = 0.3          # classes < (max_count * THRESHOLD_FACTOR) get oversampled
SYNONYM_REPLACE_COUNT = 1       # how many words to replace per augmented sample

##############################################################################
# DATA LOADING & CLEANING
##############################################################################
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    return df

def basic_clean_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # Remove email addresses
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', 'URL', text)
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace digits
    text = re.sub(r'\d+', '[NUM]', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text.strip().lower()

##############################################################################
# RARE CLASS MERGING
##############################################################################
def merge_rare_classes(df, col, min_count, merged_label="Other"):
    """
    Any class in 'col' with fewer than 'min_count' samples
    is replaced by 'merged_label'.
    """
    counts = df[col].value_counts()
    rare_classes = counts[counts < min_count].index
    df[col] = df[col].where(~df[col].isin(rare_classes), merged_label)
    return df

##############################################################################
# SYNONYM-BASED TEXT AUGMENTATION
##############################################################################
def get_synonyms(word):
    """
    Return a list of potential synonyms for a given word using WordNet.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            candidate = lemma.name().replace("_", " ").lower()
            if candidate != word:
                synonyms.add(candidate)
    return list(synonyms)

def synonym_replacement(sentence, replace_count=SYNONYM_REPLACE_COUNT):
    """
    Replace up to 'replace_count' words in 'sentence' with synonyms.
    """
    words = sentence.split()
    if not words:
        return sentence
    for _ in range(replace_count):
        idx = random.randint(0, len(words) - 1)
        syns = get_synonyms(words[idx])
        if syns:
            words[idx] = random.choice(syns)
    return " ".join(words)

##############################################################################
# OVERSAMPLING (WITH SYNONYM AUGMENTATION)
##############################################################################
def oversample_with_synonyms(df, target_col, text_col, threshold_factor=THRESHOLD_FACTOR, multiplier=OVERSAMPLE_MULTIPLIER):
    """
    For classes in target_col with < (max_count * threshold_factor) samples (but != Other),
    oversample them by creating new rows w/ synonym replacements.
    """
    counts = df[target_col].value_counts()
    max_count = counts.max()
    minority_classes = counts[counts < (max_count * threshold_factor)].index.tolist()
    
    # Exclude 'Other' from oversampling
    if "Other" in minority_classes:
        minority_classes.remove("Other")

    augmented_rows = []
    for cls in minority_classes:
        class_subset = df[df[target_col] == cls]
        for _, row in class_subset.iterrows():
            for _ in range(multiplier):
                new_row = row.copy()
                new_row[text_col] = synonym_replacement(new_row[text_col], replace_count=SYNONYM_REPLACE_COUNT)
                augmented_rows.append(new_row)

    if augmented_rows:
        df_aug = pd.DataFrame(augmented_rows)
        df = pd.concat([df, df_aug], ignore_index=True)
    return df

##############################################################################
# SAMPLE WEIGHTS (PER-SAMPLE)
##############################################################################
def compute_sample_weights(df, cat_col, subcat_col, assign_col, max_weight=10.0):
    """
    Combine frequency-based weights for Category, Subcategory,
    Assignment Group into a single sample_weight column.

    Weighted by sqrt(1/freq) for each target, then geometric-mean,
    clipped to [1.0, max_weight].
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
    
    def row_weight(row):
        w_cat = cat_weights[row[cat_col]]
        w_sub = subcat_weights[row[subcat_col]]
        w_asg = assign_weights[row[assign_col]]
        w = (w_cat * w_sub * w_asg) ** (1/3)
        return min(max_weight, max(1.0, w))
    
    return df.apply(row_weight, axis=1)

##############################################################################
# MAIN DATA PREPARATION FUNCTION
##############################################################################
def main():
    print(f"Loading data from '{EXCEL_FILE}'...")
    df = load_data(EXCEL_FILE)
    print(f"Initial shape: {df.shape}")

    # Fill missing text
    df['Short Description'] = df['Short Description'].fillna('')
    df['Description'] = df['Description'].fillna('')
    
    # Clean text
    df['Short Description'] = df['Short Description'].apply(basic_clean_text)
    df['Description'] = df['Description'].apply(basic_clean_text)
    
    # Drop rows missing targets
    target_cols = ['Category', 'Subcategory', 'Assignment Group']
    df = df.dropna(subset=target_cols)
    print(f"After dropping missing targets: {df.shape}")

    # Merge rare classes into "Other"
    df = merge_rare_classes(df, 'Category', MIN_SAMPLES_CATEGORY, 'Other')
    df = merge_rare_classes(df, 'Subcategory', MIN_SAMPLES_SUBCATEGORY, 'Other')
    df = merge_rare_classes(df, 'Assignment Group', MIN_SAMPLES_ASSIGNMENT, 'Other')

    # Fill other relevant fields for combined text
    df['Priority'] = df['Priority'].fillna('UnknownPriority')
    df['Contact Type'] = df['Contact Type'].fillna('OtherContact')
    df['Location'] = df['Location'].fillna('UnknownLocation')
    df['U Department'] = df['U Department'].fillna('UnknownDepartment')
    df['Opened By Department'] = df['Opened By Department'].fillna('UnknownDepartment')

    # Build combined text
    df['combined_text'] = (
        df['Short Description'].fillna('') + ' [SEP] ' +
        df['Description'].fillna('') + ' [SEP] ' +
        df['Priority'].astype(str) + ' [SEP] ' +
        df['Contact Type'].astype(str) + ' [SEP] ' +
        df['Location'].astype(str) + ' [SEP] ' +
        df['U Department'].astype(str) + ' [SEP] ' +
        df['Opened By Department'].astype(str)
    )

    # Oversample for each target with synonyms
    print("Performing synonym-based oversampling for Category...")
    df = oversample_with_synonyms(df, 'Category', 'combined_text', threshold_factor=THRESHOLD_FACTOR)
    print("Performing synonym-based oversampling for Subcategory...")
    df = oversample_with_synonyms(df, 'Subcategory', 'combined_text', threshold_factor=THRESHOLD_FACTOR)
    print("Performing synonym-based oversampling for Assignment Group...")
    df = oversample_with_synonyms(df, 'Assignment Group', 'combined_text', threshold_factor=THRESHOLD_FACTOR)

    print(f"Shape after oversampling: {df.shape}")

    # Compute sample weights
    df['sample_weight'] = compute_sample_weights(df, 'Category', 'Subcategory', 'Assignment Group')

    # Label encoding
    label_encoders = {}
    for col in ['Category', 'Subcategory', 'Assignment Group']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Build dictionary
    train_dict = {
        'text': df['combined_text'].tolist(),
        'category_labels': df['Category'].tolist(),
        'subcategory_labels': df['Subcategory'].tolist(),
        'assignment_group_labels': df['Assignment Group'].tolist(),
        'sample_weights': df['sample_weight'].tolist(),
        'label_encoders': label_encoders
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(train_dict, f)

    print(f"Data preparation complete. Final shape: {df.shape}")
    print(f"Saved to '{OUTPUT_PKL}'.")

if __name__ == "__main__":
    main()
