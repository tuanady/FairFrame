import modal
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Setup the App
app = modal.App("fairframe-tester")
vol = modal.Volume.from_name("fairframe-vol")

# 2. Define the Environment (Must match training)
image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "torch", "sentencepiece", "protobuf")
)

# 3. Define the Label Map (Must match training!)
ID2LABEL = {
    0: " Safe",
    1: " Gender Bias",
    2: " Racial Bias",
    3: " Age Bias",
    4: " Disability Bias",
    5: " Professional/Socioeconomic Bias"
}


@app.function(image=image, volumes={"/data": vol}, gpu="L4")
def run_tests():
    print("⬇️ Loading your trained model from volume...")
    model_path = "/data/multilabel_detector"

    # Load Model & Tokenizer
    # Note: use_fast=False is crucial for DeBERTa
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # 4. Define Test Sentences
    tests = [
        "The software engineer focused on his code.",  # Gender (assuming engineer is male)
        "He is a doctor and she is a nurse.",  # Professional + Gender
        "The old man couldn't understand how to use the iPad.",  # Age
        "The wheelchair user was an inspiration to us all.",  # Disability (Inspiration porn)
        "Drivers from that country are always aggressive.",  # Race/Nationality
        "The teacher helped the student with their homework."  # Safe
    ]

    print("\n🧐 TESTING MODEL ACCURACY:\n" + "=" * 40)

    for text in tests:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")

        # Predict
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get Winner
        pred_id = logits.argmax().item()
        score = torch.softmax(logits, dim=1)[0][pred_id].item()
        label = ID2LABEL[pred_id]

        print(f"📝 Input:  '{text}'")
        print(f"🔍 Result: {label} ({score:.1%} confidence)")
        print("-" * 40)


@app.local_entrypoint()
def main():
    run_tests.remote()