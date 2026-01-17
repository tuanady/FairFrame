from flask import Flask, render_template, request, jsonify
import modal
import os

app = Flask(__name__)

# Initialize Modal app
modal_app = modal.App("fairframe-web")
vol = modal.Volume.from_name("fairframe-vol")
image = modal.Image.debian_slim().pip_install("transformers", "torch", "sentencepiece", "protobuf")

ID2LABEL = {0: "Safe", 1: "Gender", 2: "Race", 3: "Age", 4: "Disability", 5: "Profession"}


@modal_app.function(image=image, volumes={"/data": vol}, gpu="L4")
def detect_gaps_modal(prompt: str):
    """
    Modal function to detect bias gaps in a prompt
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"🔍 Analyzing: '{prompt}'")

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

    for i in range(1, 6):  # Skip 0 (Safe)
        score = probs[i].item()
        all_scores[ID2LABEL[i]] = score
        if score > 0.50:  # Threshold
            active_gaps.append(ID2LABEL[i])

    return {
        "gaps": active_gaps,
        "scores": all_scores,
        "safe_score": probs[0].item()
    }


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to analyze a prompt"""
    import requests

    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()

        if not prompt:
            return jsonify({'error': 'Prompt cannot be empty'}), 400

        # ⭐ Call Modal web endpoint
        modal_url = "https://modal-labs-civicmachines--fairframe-tester-analyze-web.modal.run"

        response = requests.post(modal_url, json={"prompt": prompt})

        if not response.ok:
            raise Exception(f"Modal API error: {response.status_code}")

        result = response.json()
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run Flask app
    print("🚀 Starting FairFrame web server...")
    print("📍 Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)