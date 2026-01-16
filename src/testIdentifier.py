import modal
import torch
from modal_proto.api_pb2 import FUNCTION_CALL_TYPE_UNARY
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. SETUP
app = modal.App("fairframe-tester")
vol = modal.Volume.from_name("fairframe-vol")

image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "torch", "sentencepiece", "protobuf")
)

# 2. LABELS (Shortened for cleaner tags)
ID2LABEL = {
    0: "Safe",
    1: "Gender",
    2: "Race",
    3: "Age",
    4: "Disability",
    5: "Profession"
}


@app.function(image=image, volumes={"/data": vol}, gpu="L4")
def run_tests():
    print("⬇️ Loading STEM-Optimized Model...")
    model_path = "/data/multilabel_detector"

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # 3. DEFINE TESTS (Including intersectional ones)
    tests = [
        "Give me a picture of a Researcher",  # Likely Gender
        "Give me a picture of STEM student",  # Likely Race
        "Generate a math professor",  # Likely Race + Gender (Split)
        "Portrait of a successful CEO",  # Likely Gender
        "An engineer working in the field",  # Likely Disability (ableism) or Gender
        "Generate a black female researcher"  # Safe
    ]

    print("\n🔬 GAP ANALYSIS REPORT:\n" + "=" * 60)

    for text in tests:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        # Get probabilities
        probs = torch.softmax(logits, dim=1)[0]

        # 1. Total Gap Probability (Probability of NOT being Safe)
        safe_score = probs[0].item()
        gap_probability = 1.0 - safe_score

        # 2. Extract Contributors (> 10% significance threshold)
        active_gaps = []
        for i in range(1, 6):
            score = probs[i].item()
            if score > 0.10:  # If the model gives it at least 10% vote
                active_gaps.append((ID2LABEL[i], score))

        # Sort by score
        active_gaps.sort(key=lambda x: x[1], reverse=True)

        # 3. GENERATE THE DYNAMIC TAG
        if safe_score > 0.60:
            final_tag = "✅ Safe"
        elif not active_gaps:
            final_tag = "❓ Unsure"  # High gap prob, but no specific class won
        else:
            # Join the labels with slashes (e.g., "Race/Gender")
            tag_names = [x[0] for x in active_gaps]
            final_tag = "/".join(tag_names) + " Gap"

        # Output
        print(f"📝 Prompt: '{text}'")
        print(f"   -> Gap Probability: {gap_probability:.1%}")
        print(f"   -> Detected Tag:    {final_tag}")

        # Optional: Show breakdown if it's a split decision
        if len(active_gaps) > 1:
            details = ", ".join([f"{n} ({s:.1%})" for n, s in active_gaps])
            print(f"   -> Split Details:   {details}")

        print("-" * 60)


@app.local_entrypoint()
def main():
    run_tests.remote()
