from __future__ import annotations
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def predict_hedonic_labels(df, model, tokenizer, device, hedonic_labels, column_FreeJAR_description: int, hypothesis_template: str = None):
    """
    Predict hedonic labels for each row in the DataFrame using NLI (natural language inference).

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        model: Hugging Face transformer model (e.g., BERT for NLI).
        tokenizer: Corresponding tokenizer for the model.
        device: 'cpu' or 'cuda'.
        hedonic_labels (list of str): List of hedonic labels.
        column_index (int): Index of the column containing the premise text.
        hypothesis_template (str, optional): Template string with one placeholder. 
                                             Defaults to "The consumer of this product said {}."

    Returns:
        pd.DataFrame: Original DataFrame with added probability columns and a 'predict' column.
    """
    # Get column name from index
    try:
        text_column = df.columns[column_FreeJAR_description]
    except IndexError:
        raise ValueError(f"Invalid column_index: {column_FreeJAR_description}. DataFrame has {len(df.columns)} columns.")

    # Default hypothesis template
    if hypothesis_template is None:
        hypothesis_template = "The consumer of this product said {}."

    hypotheses = [hypothesis_template.format(label) for label in hedonic_labels]
    results = []

    for premise in df[text_column].tolist():
        pairs = [(premise, hypothesis) for hypothesis in hypotheses]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        entailment_probs = torch.softmax(logits, dim=0)[:, 0]  # Index 0: entailment
        results.append(entailment_probs)

    df_probs = pd.DataFrame([t.tolist() for t in results], columns=hedonic_labels)
    df_combined = pd.concat([df.reset_index(drop=True), df_probs], axis=1)
    df_combined['predict'] = df_combined[hedonic_labels].idxmax(axis=1)
    df_combined['consistency'] = df_combined.apply(lambda row: 'consistent' if row['Hedonic_category'] == row['predict'] else 'inconsistent', axis=1)
    df_combined["wrong_pred"] = df_combined['Hedonic_category'] != df_combined['predict']
    df_combined["serious_error"] = ((df_combined['Hedonic_category'] == "I don't like it") & (df_combined['predict'] == "I like it a lot")) | ((df_combined['Hedonic_category'] == "I like it a lot") & (df_combined['predict'] == "I don't like it"))
    df_combined["morderate_error"] = ((df_combined['Hedonic_category'] == "I like it moderately") & ((df_combined['predict'] == "I like it a lot") | (df_combined['predict'] == "I don't like it")))
    return df_combined

def assess_consistency(df, model, tokenizer, device, hedonic_labels, column_FreeJAR_description: int, hypothesis_template: str = None, batch_size: int = 16):
    try:
        text_column = df.columns[column_FreeJAR_description]
    except IndexError:
        raise ValueError(f"Invalid column_index: {column_FreeJAR_description}. DataFrame has {len(df.columns)} columns.")

    if hypothesis_template is None:
        hypothesis_template = "The consumer of this product said {}."

    hypotheses = [hypothesis_template.format(label) for label in hedonic_labels]
    premises = df[text_column].tolist()
    num_premises = len(premises)
    results = []

    # tqdm hiển thị progress bar khi lặp theo batch
    for i in tqdm(range(0, num_premises, batch_size), desc="Processing", unit="batch"):
        batch_premises = premises[i:i+batch_size]
        batch_pairs = [(premise, hyp) for premise in batch_premises for hyp in hypotheses]

        inputs = tokenizer(batch_pairs, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        logits = logits.view(len(batch_premises), len(hypotheses), -1)
        entailment_probs = torch.softmax(logits, dim=2)[:, :, 0]
        results.extend(entailment_probs.cpu())

    # Kết quả ra DataFrame
    df_probs = pd.DataFrame([t.tolist() for t in results], columns=hedonic_labels)
    df_combined = pd.concat([df.reset_index(drop=True), df_probs], axis=1)

    # Predict & consistency
    df_combined['predict'] = df_combined[hedonic_labels].idxmax(axis=1)
    df_combined['consistency'] = df_combined.apply(
        lambda row: 'consistent' if row['Hedonic_category'] == row['predict'] else 'inconsistent', axis=1)
    df_combined['wrong_pred'] = df_combined['Hedonic_category'] != df_combined['predict']
    df_combined['serious_error'] = (
        ((df_combined['Hedonic_category'] == "I don't like it") & (df_combined['predict'] == "I like it a lot")) |
        ((df_combined['Hedonic_category'] == "I like it a lot") & (df_combined['predict'] == "I don't like it"))
    )
    df_combined['morderate_error'] = (
        ((df_combined['Hedonic_category'] == "I like it moderately") &
         ((df_combined['predict'] == "I like it a lot") | (df_combined['predict'] == "I don't like it")))
    )

    return df_combined
