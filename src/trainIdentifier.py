import modal
from pathlib import Path

app = modal.App("fairframe-multilabel-sigmoid")

# 1. SETUP PATHS
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent

image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "datasets", "torch", "scikit-learn", "accelerate", "pandas", "scipy", "sentencepiece",
                 "protobuf")
    # Mount BOTH datasets
    .add_local_file(project_root / "generatedDataset.csv", "/root/generatedDataset.csv")
    .add_local_file(project_root / "cpDataset.csv", "/root/cpDataset.csv")
)

vol = modal.Volume.from_name("fairframe-vol", create_if_missing=True)


@app.function(image=image, gpu="L4", timeout=3600, volumes={"/data": vol})
def train_multilabel():
    import pandas as pd
    import json
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import torch
    import numpy as np

    print("REMOTE FUNCTION STARTED: Training Hybrid Model")

    # --- 1. DEFINE LABELS ---
    # 0=Safe, 1=Gender, 2=Race, 3=Age, 4=Disability, 5=Profession
    LABEL_MAP = {"safe": 0, "gender": 1, "race": 2, "age": 3, "disability": 4, "profession": 5}
    ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
    NUM_LABELS = len(LABEL_MAP)

    texts = []
    labels = []  # List of multi-hot vectors

    # --- 2. LOAD DATA ---
    print("Loading Data...")

    # A. Load The Generated Dataset
    # This detects "Missing" diversity
    try:
        df_gen = pd.read_csv("/root/generatedDataset.csv")
        print(f"   Found generatedDataset with {len(df_gen)} rows.")

        BOOST_FACTOR = 20  # Keep your high weighting for Gaps

        for _, row in df_gen.iterrows():
            # Parse the JSON-encoded label vector directly (Don't touch logic)
            label_vector = json.loads(row['labels'])

            # Add boosted copies
            for _ in range(BOOST_FACTOR):
                texts.append(str(row['text']))
                labels.append(label_vector)

        print(f"Loaded Gap Data: {len(df_gen)} examples -> boosted to {len(df_gen) * BOOST_FACTOR}.")

    except Exception as e:
        print(f"Error loading generatedDataset: {e}")
        return

    # B. Load CrowS-Pairs
    print("   Loading CrowS-Pairs...")
    try:
        df_crows = pd.read_csv("/root/cpDataset.csv")
        crows_count = 0

        for _, row in df_crows.iterrows():
            # 1. Map text category to our ID
            bias_type_raw = str(row['bias_type']).lower()

            # Default to 'Profession' (5) if ambiguous
            target_idx = 5
            if 'gender' in bias_type_raw:
                target_idx = 1
            elif 'race' in bias_type_raw:
                target_idx = 2
            elif 'disability' in bias_type_raw:
                target_idx = 4
            elif 'age' in bias_type_raw:
                target_idx = 3

            # 2. Add Biased Sentence ("Sent More") -> Specific Bias Vector
            # We create the vector manually: [0, 0, 1, 0, 0, 0]
            vec_bias = [0.0] * NUM_LABELS
            vec_bias[target_idx] = 1.0

            texts.append(str(row['sent_more']))
            labels.append(vec_bias)

            # 3. Add Safe Sentence ("Sent Less") -> Safe Vector
            # We create the vector manually: [1, 0, 0, 0, 0, 0]
            vec_safe = [0.0] * NUM_LABELS
            vec_safe[0] = 1.0

            texts.append(str(row['sent_less']))
            labels.append(vec_safe)

            crows_count += 1

        print(f" Loaded CrowS-Pairs: {crows_count} pairs -> {crows_count * 2} training instances.")

    except Exception as e:
        print(f" Warning: Could not load CrowS-Pairs: {e}")

    # --- 3. VERIFY DATA MIX ---
    print("\n Data Distribution Check:")
    print(f"   Total Training Samples: {len(texts)}")

    # Simple check to ensure both types exist
    label_counts = [0] * NUM_LABELS
    for label_vec in labels:
        for i, val in enumerate(label_vec):
            if val > 0:
                label_counts[i] += 1

    print("   Class Breakdown:")
    for i, count in enumerate(label_counts):
        print(f"   - {ID2LABEL[i]}: {count} ({count / len(labels) * 100:.1f}%)")

    # --- 4. TRAIN ---
    MODEL_NAME = "microsoft/deberta-v3-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
        problem_type="multi_label_classification"
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42
    )

    class MultiLabelDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            # Labels must be FLOAT for Sigmoid (BCE) loss
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item

        def __len__(self):
            return len(self.labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = MultiLabelDataset(train_encodings, train_labels)
    val_dataset = MultiLabelDataset(val_encodings, val_labels)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/results",
            num_train_epochs=5,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            save_strategy="no",
            eval_strategy="epoch",
            logging_steps=50,
            report_to="none"
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("\n Starting Training...")
    trainer.train()

    save_path = "/data/multilabel_detector"
    print(f"\nSaving Hybrid Model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Training Complete!")


@app.local_entrypoint()
def main():
    train_multilabel.remote()