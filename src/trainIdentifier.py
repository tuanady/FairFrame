import modal
import sys
from pathlib import Path

app = modal.App("fairframe-multilabel-perfect")

# 1. SETUP PATHS
# We assume this script is in src/, so we go up one level to find the CSVs
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent

# 2. DEFINE IMAGE & MOUNTS
image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "datasets", "torch", "scikit-learn", "accelerate", "pandas", "scipy", "sentencepiece",
                 "protobuf")
    # Mount files from the project root into the cloud container
    .add_local_file(project_root / "crows_pairs_anonymized.csv", "/root/crows_pairs_anonymized.csv")
    .add_local_file(project_root / "augmented_bias.csv", "/root/augmented_bias.csv")
)

vol = modal.Volume.from_name("fairframe-vol", create_if_missing=True)


@app.function(image=image, gpu="L4", timeout=3600, volumes={"/data": vol})
def train_multiclass():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import torch
    import numpy as np

    print("🚀 REMOTE FUNCTION STARTED: Training 'Perfect' Model...")

    # --- 1. DEFINE LABELS ---
    LABEL_MAP = {"safe": 0, "gender": 1, "race": 2, "age": 3, "disability": 4, "profession": 5}
    ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

    texts = []
    labels = []

    # --- 2. LOAD DATA ---
    print("1️⃣  Loading CSVs from /root/...")

    # Load Original CrowS-Pairs
    try:
        df1 = pd.read_csv("/root/crows_pairs_anonymized.csv")
        for _, row in df1.iterrows():
            bias_type_raw = str(row['bias_type']).lower()
            cat_id = 5
            if 'gender' in bias_type_raw:
                cat_id = 1
            elif 'race' in bias_type_raw:
                cat_id = 2
            elif 'age' in bias_type_raw:
                cat_id = 3
            elif 'disability' in bias_type_raw:
                cat_id = 4
            elif 'socioeconomic' in bias_type_raw:
                cat_id = 5

            texts.append(str(row['sent_more']))
            labels.append(cat_id)
            texts.append(str(row['sent_less']))
            labels.append(0)
    except Exception as e:
        print(f"❌ Error loading CrowS: {e}")

    # Load Augmented Data
    try:
        df2 = pd.read_csv("/root/augmented_bias.csv")
        for _, row in df2.iterrows():
            # 3x Weighting for Augmented Data
            for _ in range(3):
                texts.append(str(row['sent_more']))
                labels.append(int(row['bias_type_id']))
            for _ in range(3):
                texts.append(str(row['sent_less']))
                labels.append(0)
        print(f"✅ Added {len(df2) * 3} boosted examples.")
    except Exception as e:
        print(f"❌ Error loading Augmented Data: {e}")

    # --- 3. TRAIN ---
    MODEL_NAME = "microsoft/deberta-v3-base"
    print(f"⚙️ Loading Tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6,
        id2label=ID2LABEL,
        label2id=LABEL_MAP
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.15, random_state=42)

    class MultiBiasDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = MultiBiasDataset(train_encodings, train_labels)
    val_dataset = MultiBiasDataset(val_encodings, val_labels)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/results",
            num_train_epochs=5,
            per_device_train_batch_size=16,
            eval_strategy="epoch",
            learning_rate=2e-5,
            save_strategy="no",
            report_to="none"
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: {"accuracy": (np.argmax(p.predictions, axis=1) == p.label_ids).mean()}
    )

    trainer.train()

    save_path = "/data/multilabel_detector"
    print(f"💾 Saving Perfect Model to {save_path}...")
    model.save_pretrained(save_path)