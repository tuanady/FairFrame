import modal
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = modal.App("fairframe-tester")
vol = modal.Volume.from_name("fairframe-vol")
image = modal.Image.debian_slim().pip_install(
    "transformers",
    "torch",
    "sentencepiece",
    "protobuf",
    "fastapi[standard]"
)

ID2LABEL = {0: "Safe", 1: "Gender", 2: "Race", 3: "Age", 4: "Disability"}


@app.function(image=image, volumes={"/data": vol}, gpu="L4")
def run_tests():
    print(" Loading Sigmoid Model...")
    model_path = "/data/multilabel_detector"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    tests = [
        "Give me a picture of a Researcher",
        "Give me a picture of STEM student",
        "Generate a math professor",
        "Generate a black female researcher",
        "Generate a picture of a male chemist",
        "Give me a picture of a white engineer",
    ]

    print("\n INDEPENDENT GAP ANALYSIS:\n" + "=" * 60)

    for text in tests:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        # USE SIGMOID (Independent Probabilities 0-100% for EACH class)
        probs = torch.sigmoid(logits)[0]

        # 1. Check Safe Score
        safe_score = probs[0].item()

        # 2. Check Gaps (Independent Thresholds)
        active_gaps = []
        for i in range(1, 6):
            score = probs[i].item()
            # If > 50%, we consider it a detected gap
            if score > 0.50:
                active_gaps.append((ID2LABEL[i], score))

        # Sort just for display
        active_gaps.sort(key=lambda x: x[1], reverse=True)

        if safe_score > 0.80 and not active_gaps:
            final_tag = "✅ Safe"
        elif not active_gaps:
            final_tag = "❓ Low Confidence"
        else:
            tag_names = [x[0] for x in active_gaps]
            final_tag = "/".join(tag_names) + " Gap"

        print(f"📝 Prompt: '{text}'")
        print(f"   -> Detected Tag:    {final_tag}")

        if active_gaps:
            details = ", ".join([f"{n} ({s:.1%})" for n, s in active_gaps])
            print(f"   -> Scores:          {details}")
        print("-" * 60)


@app.function(image=image, volumes={"/data": vol}, gpu="L4")
@modal.web_endpoint(method="POST")
def analyze_web(data: dict):
    """
    Web endpoint for bias detection - callable via HTTP from Flask
    """
    prompt = data.get("prompt", "")

    print(f" Analyzing via web: '{prompt}'")

    # Load model
    model_path = "/data/multilabel_detector"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Get predictions
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.sigmoid(logits)[0]

    # Detect active gaps
    active_gaps = []
    all_scores = {}

    for i in range(1, 6):
        score = probs[i].item()
        all_scores[ID2LABEL[i]] = score
        if score > 0.50:
            active_gaps.append(ID2LABEL[i])

    return {
        "gaps": active_gaps,
        "scores": all_scores,
        "safe_score": probs[0].item()
    }


@app.local_entrypoint()
def main():
    run_tests.remote()

