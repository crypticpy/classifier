#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit app hosting the DeBERTa-v3 multi-task model for ServiceNow tickets.
Predicts:
    - Category
    - Subcategory
    - Assignment Group

Expects:
    - best_deberta_v3_weights.pt (OR a checkpoint) for the model
    - label_encoders_only.pkl (for label encoders, minimal data)

Run with:
    streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import re
import nltk
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd

# Ensure necessary NLTK downloads (if not pre-installed)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


##############################################################################
# MODEL DEFINITION
##############################################################################
class DebertaV3MultiTask(nn.Module):
    """
    Multi-task model using DeBERTa-v3-base as a shared encoder,
    plus 3 classification heads for Category, Subcategory, Assignment Group.
    """
    def __init__(self, model_name, num_categories, num_subcategories, num_assignment_groups):
        super(DebertaV3MultiTask, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.category_classifier = nn.Linear(hidden_size, num_categories)
        self.subcategory_classifier = nn.Linear(hidden_size, num_subcategories)
        self.assignment_group_classifier = nn.Linear(hidden_size, num_assignment_groups)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        logits_category = self.category_classifier(pooled_output)
        logits_subcategory = self.subcategory_classifier(pooled_output)
        logits_assignment_group = self.assignment_group_classifier(pooled_output)
        return logits_category, logits_subcategory, logits_assignment_group


##############################################################################
# PREPROCESSING (MATCH TRAINING PIPELINE)
##############################################################################
def basic_clean_text(text):
    """
    Light cleaning to mimic your EDA/data prep:
      - Remove emails
      - Remove URLs
      - Remove special chars
      - Mask digits
      - Normalize spacing
      - Lowercase
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    text = re.sub(r'http\S+|www.\S+', 'URL', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '[NUM]', text)
    text = ' '.join(text.split())
    return text.lower().strip()


def build_combined_text(short_desc, description, priority, contact_type,
                        location, u_department, opened_by_dept):
    """
    Reproduce how text was combined in your data prep:
      [Short Description] [SEP] [Description] [SEP] [Priority] [SEP] ...
    """
    short_desc = basic_clean_text(short_desc)
    description = basic_clean_text(description)
    priority = basic_clean_text(priority)
    contact_type = basic_clean_text(contact_type)
    location = basic_clean_text(location)
    u_department = basic_clean_text(u_department)
    opened_by_dept = basic_clean_text(opened_by_dept)

    combined = (f"{short_desc} [SEP] {description} [SEP] {priority} [SEP] "
                f"{contact_type} [SEP] {location} [SEP] {u_department} [SEP] "
                f"{opened_by_dept}")
    return combined


##############################################################################
# STREAMLIT APP
##############################################################################
def main():
    st.title("ServiceNow Multi-Task Classifier (DeBERTa-v3)")
    st.write("Predicts Category → Subcategory → Assignment Group from ticket fields.")

    # --------------------------------------------------------
    # 1) LOAD LABEL ENCODERS
    # --------------------------------------------------------
    with open("label_encoders_only.pkl", "rb") as f:
        label_info = pickle.load(f)
    label_encoders = label_info["label_encoders"]

    le_category = label_encoders["Category"]
    le_subcategory = label_encoders["Subcategory"]
    le_assignment = label_encoders["Assignment Group"]

    num_categories = label_info["num_categories"]
    num_subcategories = label_info["num_subcategories"]
    num_assignment_groups = label_info["num_assignment_groups"]

    # --------------------------------------------------------
    # 2) MODEL INIT
    # --------------------------------------------------------
    model_name = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = DebertaV3MultiTask(
        model_name=model_name,
        num_categories=num_categories,
        num_subcategories=num_subcategories,
        num_assignment_groups=num_assignment_groups
    )

    # --------------------------------------------------------
    # 3) LOAD MODEL WEIGHTS OR CHECKPOINT
    # --------------------------------------------------------
    checkpoint_path = "best_deberta_v3_weights_fixed.pt"  # or e.g. "checkpoint_epoch_9.pt"

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # If it's a full checkpoint dict, it will have "model_state_dict".
        # If it's a raw model state, just load it directly.
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        st.write(f"Loaded model weights from: {checkpoint_path}")
    except Exception as e:
        st.error(f"Error loading weights from '{checkpoint_path}': {e}")
        st.stop()

    model.eval()

    # For Apple Silicon M1/M2/M3 or GPU if available:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)

    # --------------------------------------------------------
    # 4) STREAMLIT INPUT FIELDS
    # --------------------------------------------------------
    st.subheader("Enter Incident ID or Ticket Fields")
    incident_id = st.text_input("Incident ID (e.g., INC0360850)", "")
    
    # Load data button
    if st.button("Load Incident Data"):
        if incident_id:
            try:
                df = pd.read_excel("test_data.xlsx")  # Replace with your file path
                incident_data = df[df['Number Fld'] == incident_id].iloc[0]
                
                if not incident_data.empty:
                    st.session_state.short_desc = incident_data['Short Description']
                    st.session_state.description = incident_data['Description']
                    st.session_state.priority = incident_data['Priority']
                    st.session_state.contact_type = incident_data['Contact Type']
                    st.session_state.location = incident_data['Location']
                    st.session_state.u_dept = incident_data['U Department']
                    st.session_state.opened_by_dept = incident_data['Opened By Department']
                    st.success(f"Data loaded for Incident ID: {incident_id}")
                else:
                    st.error(f"Incident ID '{incident_id}' not found in the Excel file.")
            except FileNotFoundError:
                st.error("Excel file not found. Please ensure 'your_excel_file.xlsx' is in the same directory.")
            except Exception as e:
                st.error(f"Error loading data: {e}")
        else:
            st.warning("Please enter an Incident ID.")
    
    # Initialize session state for input fields if not already set
    if 'short_desc' not in st.session_state:
        st.session_state.short_desc = ""
    if 'description' not in st.session_state:
        st.session_state.description = ""
    if 'priority' not in st.session_state:
        st.session_state.priority = ""
    if 'contact_type' not in st.session_state:
        st.session_state.contact_type = ""
    if 'location' not in st.session_state:
        st.session_state.location = ""
    if 'u_dept' not in st.session_state:
        st.session_state.u_dept = ""
    if 'opened_by_dept' not in st.session_state:
        st.session_state.opened_by_dept = ""

    short_desc = st.text_input("Short Description", st.session_state.short_desc)
    description = st.text_area("Description", st.session_state.description)
    priority = st.text_input("Priority", st.session_state.priority)
    contact_type = st.text_input("Contact Type", st.session_state.contact_type)
    location = st.text_input("Location", st.session_state.location)
    u_dept = st.text_input("U Department", st.session_state.u_dept)
    opened_by_dept = st.text_input("Opened By Department", st.session_state.opened_by_dept)

    # --------------------------------------------------------
    # 5) INFERENCE
    # --------------------------------------------------------
    if st.button("Predict"):
        combined_text = build_combined_text(
            short_desc, description, priority, contact_type,
            location, u_dept, opened_by_dept
        )

        inputs = tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            logits_cat, logits_subcat, logits_assign = model(input_ids, attention_mask)

        pred_cat_id = torch.argmax(logits_cat, dim=1).cpu().item()
        pred_subcat_id = torch.argmax(logits_subcat, dim=1).cpu().item()
        pred_assign_id = torch.argmax(logits_assign, dim=1).cpu().item()

        pred_category = le_category.inverse_transform([pred_cat_id])[0]
        pred_subcategory = le_subcategory.inverse_transform([pred_subcat_id])[0]
        pred_assignment = le_assignment.inverse_transform([pred_assign_id])[0]

        st.markdown(f"**Predicted Category**: {pred_category}")
        st.markdown(f"**Predicted Subcategory**: {pred_subcategory}")
        st.markdown(f"**Predicted Assignment Group**: {pred_assignment}")


if __name__ == "__main__":
    main()