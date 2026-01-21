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

    print("🚀 REMOTE FUNCTION STARTED: Training Improved Hybrid Model")

    # --- 1. DEFINE LABELS ---
    LABEL_MAP = {"safe": 0, "gender": 1, "race": 2, "age": 3, "disability": 4, "profession": 5}
    ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
    NUM_LABELS = len(LABEL_MAP)

    texts = []
    labels = []  # List of multi-hot vectors

    # ⭐ NEW: Label Smoothing Function
    def apply_label_smoothing(label_vector, smoothing=0.1):
        """
        Apply label smoothing to reduce overconfidence
        1.0 -> 0.95, 0.0 -> 0.025
        """
        smoothed = []
        for val in label_vector:
            if val == 1.0:
                smoothed.append(1.0 - smoothing)
            else:
                smoothed.append(smoothing / (NUM_LABELS * 2))
        return smoothed

    # ⭐ NEW: Text Augmentation Function
    def augment_text(text, num_variations=3):
        """
        Create slight variations to reduce exact memorization
        """
        variations = [text]  # Original

        # Add variation prefixes
        prefixes = ["", "Show me ", "Generate ", "Create ", "Give me a picture of "]
        articles = ["a ", "an ", "the ", ""]

        # Extract base text (remove common prefixes)
        base = text.lower().strip()
        for prefix in ["show me ", "generate ", "create ", "give me a picture of ", "a ", "an ", "the "]:
            if base.startswith(prefix):
                base = base[len(prefix):].strip()

        # Generate variations
        import random
        for _ in range(min(num_variations, 3)):
            prefix = random.choice(prefixes)
            article = random.choice(articles)

            if prefix:
                variations.append(f"{prefix}{article}{base}")
            else:
                variations.append(f"{article}{base}")

        return list(set(variations))[:num_variations + 1]

    # --- 2. LOAD DATA ---
    print("📊 Loading Data...")

    # A. Load The Generated Dataset (Gap Detection)
    # ... inside train_multilabel() ...

    # A. Load The Generated Dataset (Gap Detection)
    try:
        df_gen = pd.read_csv("/root/generatedDataset.csv")
        print(f"   ✓ Found generatedDataset with {len(df_gen)} rows.")

        # 🚀 ADD THIS BACK!
        # Boost gap examples so they aren't drowned out by the Safe/Bias examples.
        BOOST_FACTOR = 10

        for _, row in df_gen.iterrows():
            # Parse the JSON-encoded label vector
            label_vector = json.loads(row['labels'])

            # Apply label smoothing
            smoothed_labels = apply_label_smoothing(label_vector, smoothing=0.1)

            # Generate text variations
            # We also loop BOOST_FACTOR times to repeat the data
            text_variations = augment_text(str(row['text']), num_variations=3)

            for _ in range(BOOST_FACTOR):  # <--- CRITICAL LOOP
                for text_var in text_variations:
                    texts.append(text_var)
                    labels.append(smoothed_labels)

        print(f"   ✓ Gap Data: {len(df_gen)} rows x {BOOST_FACTOR} boost = {len(texts)} training samples.")

    except Exception as e:
        print(f"   ❌ Error loading generatedDataset: {e}")
        return

    # B. Load CrowS-Pairs (Bias Detection)
    print("   📚 Loading CrowS-Pairs...")
    try:
        df_crows = pd.read_csv("/root/cpDataset.csv")
        crows_count = 0

        for _, row in df_crows.iterrows():
            # 1. Map bias type to our label
            bias_type_raw = str(row['bias_type']).lower()

            target_idx = 5  # Default to 'Profession'
            if 'gender' in bias_type_raw:
                target_idx = 1
            elif 'race' in bias_type_raw:
                target_idx = 2
            elif 'disability' in bias_type_raw:
                target_idx = 4
            elif 'age' in bias_type_raw:
                target_idx = 3

            # 2. Add Biased Sentence with smoothing
            vec_bias = [0.0] * NUM_LABELS
            vec_bias[target_idx] = 1.0
            smoothed_bias = apply_label_smoothing(vec_bias, smoothing=0.1)

            texts.append(str(row['sent_more']))
            labels.append(smoothed_bias)

            # 3. Add Safe Sentence with smoothing
            vec_safe = [0.0] * NUM_LABELS
            vec_safe[0] = 1.0
            smoothed_safe = apply_label_smoothing(vec_safe, smoothing=0.1)

            texts.append(str(row['sent_less']))
            labels.append(smoothed_safe)

            crows_count += 1

        print(f"   ✓ CrowS-Pairs: {crows_count} pairs -> {crows_count * 2} instances.")

    except Exception as e:
        print(f"   ⚠️  Could not load CrowS-Pairs: {e}")

    # --- 3. VERIFY DATA MIX ---
    print("\n📈 Data Distribution Check:")
    print(f"   Total Training Samples: {len(texts)}")

    # Check for duplicates
    unique_texts = set(texts)
    print(f"   Unique Texts: {len(unique_texts)}")
    print(f"   Duplication Factor: {len(texts) / len(unique_texts):.1f}x")

    # Class breakdown
    label_counts = [0] * NUM_LABELS
    for label_vec in labels:
        for i, val in enumerate(label_vec):
            if val > 0.5:  # Count as active if > 0.5 after smoothing
                label_counts[i] += 1

    print("\n   Class Breakdown:")
    for i, count in enumerate(label_counts):
        print(f"     • {ID2LABEL[i]}: {count} ({count / len(labels) * 100:.1f}%)")

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

    print(f"\n   Train/Val Split: {len(train_texts)} / {len(val_texts)}")

    class MultiLabelDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item

        def __len__(self):
            return len(self.labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = MultiLabelDataset(train_encodings, train_labels)
    val_dataset = MultiLabelDataset(val_encodings, val_labels)

    # ⭐ IMPROVED TRAINING ARGUMENTS
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/results",
            num_train_epochs=4,  # ⭐ Reduced from 5
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,  # ⭐ Added regularization
            warmup_steps=100,  # ⭐ Gradual learning rate warmup
            save_strategy="no",
            eval_strategy="epoch",
            logging_steps=50,
            load_best_model_at_end=False,
            report_to="none"
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    print("\n🏋️  Starting Training...\n")
    trainer.train()

    save_path = "/data/multilabel_detector"
    print(f"\n💾 Saving Improved Model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("✅ Training Complete!")


@app.local_entrypoint()
def main():
    train_multilabel.remote()