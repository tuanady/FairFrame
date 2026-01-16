import modal
import sys
from pathlib import Path

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# Define volume to store the trained model
vol = modal.Volume.from_name("fairframe-vol", create_if_missing=True)

# Define the environment (Image)
image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers", 
        "datasets", 
        "torch", 
        "scikit-learn", 
        "accelerate", 
        "pandas", 
        "scipy", 
        "sentencepiece",
        "protobuf",
        "streamlit"
    )
)

app = modal.App("fairframe-unified")

# Label Mapping (Must match training and inference)
LABEL_MAP = {"safe": 0, "gender": 1, "race": 2, "age": 3, "disability": 4, "profession": 5}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

# UI Display Mapping
UI_LABEL_MAP = {
    "gender": "This text reinforces a gender role stereotype",
    "race": "This text contains racial or ethnic bias",
    "age": "This text marginalizes people based on age",
    "disability": "This text assumes ability or excludes disability",
    "safe": "This text is neutral and inclusive",
    "profession": "This text contains socioeconomic or professional bias"
}

# ==========================================
# 2. TRAINING FUNCTION
# ==========================================
# Run this once via: modal run fairframe_app.py::train_multiclass
@app.function(image=image, gpu="L4", timeout=3600, volumes={"/data": vol})
def train_multiclass():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    import torch
    import numpy as np

    print("🚀 REMOTE FUNCTION STARTED: Training 'Perfect' Model...")
    
    # --- MOCK DATA GENERATION (Replace this block with your actual CSV loading) ---
    # If using real files, ensure you add .add_local_file(...) to the image definition above
    print("⚠️ Using mock data. To use real data, mount your CSVs in the image definition.")
    texts = ["He is a doctor", "She is a nurse", "A person working", "Legacy admission"] * 50
    labels = [1, 1, 0, 5] * 50 
    # ----------------------------------------------------------------------------

    MODEL_NAME = "microsoft/deberta-v3-base"
    
    # Load base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=6, id2label=ID2LABEL, label2id=LABEL_MAP
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.15)

    class MultiBiasDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self): return len(self.labels)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/results", 
            num_train_epochs=1, 
            per_device_train_batch_size=8,
            save_strategy="no" # We save manually at the end
        ),
        train_dataset=MultiBiasDataset(train_encodings, train_labels),
        eval_dataset=MultiBiasDataset(val_encodings, val_labels),
    )
    
    trainer.train()

    save_path = "/data/multilabel_detector"
    print(f"💾 Saving Model to Volume at {save_path}...")
    
    # Save Model Weights
    model.save_pretrained(save_path)
    # We do NOT save the tokenizer here to avoid the 'spm.model' path issue.
    # We will load the tokenizer from the Hub during inference.
    
    vol.commit() 
    print("✅ Training complete and model saved.")

# ==========================================
# 3. INFERENCE CLASS (The "Back-end")
# ==========================================
@app.cls(image=image, gpu="L4", volumes={"/data": vol})
class BiasPredictor:
    def enter(self):
        """Loads the model once when the container starts."""
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        import os
        
        # 1. Load Tokenizer from HUB (Fixes the TypeError)
        # We always use the base tokenizer because we didn't train a new vocabulary
        base_model = "microsoft/deberta-v3-base"
        print(f"⚙️ Loading Tokenizer from {base_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

        # 2. Load Model Weights from VOLUME
        model_path = "/data/multilabel_detector"
        print(f"⚙️ Loading Weights from {model_path}...")
        
        try:
            if os.path.exists(model_path):
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                print("✅ Custom model loaded.")
            else:
                raise FileNotFoundError("Model files not found in volume.")
        except Exception as e:
            print(f"❌ Error loading custom model: {e}")
            print("⚠️ Falling back to base model (untrained) for debugging.")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                base_model, num_labels=6, id2label=ID2LABEL, label2id=LABEL_MAP
            )

        self.model.eval()
        self.model.to("cuda")

    @modal.method()
    def predict(self, text):
        import torch
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to("cuda")
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Format Results
        scores = probs.cpu().numpy()[0]
        results = []
        for idx, score in enumerate(scores):
            label_key = ID2LABEL.get(idx, "unknown")
            ui_label = UI_LABEL_MAP.get(label_key, label_key)
            results.append({"label": ui_label, "score": float(score)})
            
        return results

# ==========================================
# 4. STREAMLIT APP (The "Front-end")
# ==========================================
@app.function(image=image, allow_concurrent_inputs=100)
@modal.web_server(8501)
def run_streamlit():
    import os
    import sys
    import subprocess
    
    # The actual Streamlit Python script written to a local file in the container
    streamlit_script = """
import streamlit as st
import pandas as pd
import random
from modal import Function

st.set_page_config(page_title="⚖️ FairPrompt", layout="wide")
st.title("⚖️ FairPrompt — Custom DeBERTa Model")

ATTR_ETHNICITY = ["Indigenous", "South Asian", "Black", "Latinx", "East Asian"]
ATTR_GENDER = ["non-binary person", "woman", "man"]

def generate_fair_prompt(prompt):
    e = random.choice(ATTR_ETHNICITY)
    g = random.choice(ATTR_GENDER)
    return f"A realistic photo of a {g} {e} as a {prompt.lower()}."

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    bias_threshold = st.slider("Bias Threshold", 0.05, 1.0, 0.20)

# --- MAIN UI ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Input Prompt")
    user_input = st.text_input("Prompt:", "A portrait of a hardworking cleaner")

    if st.button("Run Analysis", type="primary"):
        with st.spinner("Querying Modal Inference Layer..."):
            try:
                # Call the remote Modal class
                # We use Function.lookup to find the class method by name
                bias_predictor = Function.lookup("fairframe-unified", "BiasPredictor.predict")
                results = bias_predictor.remote(user_input)
                st.session_state.results = results
            except Exception as e:
                st.error(f"Connection Error: {e}")

with col_right:
    if "results" in st.session_state:
        res = st.session_state.results
        
        # Sort by score
        res.sort(key=lambda x: x["score"], reverse=True)
        
        st.subheader("2. Model Predictions")
        
        # Display as table
        df = pd.DataFrame(res)
        st.dataframe(df.style.format({"score": "{:.2%}"}))

        # Check Thresholds
        risks = [r for r in res if r["score"] > bias_threshold and "neutral" not in r["label"]]
        
        st.divider()
        st.subheader("3. Findings")
        
        if risks:
            for r in risks:
                st.warning(f"⚠️ **{r['label']}** ({int(r['score']*100)}%)")
            
            st.info("Try this inclusive alternative:")
            st.success(generate_fair_prompt(user_input))
        else:
            st.success("✅ No significant bias detected.")
"""
    # Write the script to a file
    with open("app_ui.py", "w", encoding="utf-8") as f:
        f.write(streamlit_script)

    # Run Streamlit
    subprocess.call(["streamlit", "run", "app_ui.py", "--server.port=8501", "--server.address=0.0.0.0"])