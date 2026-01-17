import modal
import random
import json
from pathlib import Path

# 1. SETUP PATHS
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent

# 2. SETUP IMAGE & MOUNT JSON
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "scipy",
        "fastapi[standard]"
    )
    # MOUNT THE JSON FILE
    .add_local_file(project_root / "demographics.json", "/root/demographics.json")
)

app = modal.App("fairframe-rewriter")


@app.cls(image=image, gpu="L4", timeout=600, scaledown_window=300)
class Rewriter:

    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # 1. Load Demographics from JSON
        print("⬇️ Loading Demographics JSON...")
        try:
            with open("/root/demographics.json", "r") as f:
                self.demographics = json.load(f)
            print(f"✅ Loaded Options: {list(self.demographics.keys())}")
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            raise e

        # 2. Load Model
        print("⬇️ Loading Llama-3 (4-bit)...")
        model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                load_in_4bit=True,
            )
            print("✅ Llama-3 Loaded.")
        except Exception as e:
            print(f"❌ Error loading Model: {e}")
            raise e

    def _get_random_attributes(self, gaps):
        """
        Picks attributes from the loaded JSON.
        """
        selected_traits = []

        # Check each gap type against our JSON keys
        for gap in ["Race", "Gender", "Disability", "Age"]:
            if gap in gaps and gap in self.demographics:
                options = self.demographics[gap]
                if options:
                    trait = random.choice(options)
                    selected_traits.append(trait)

        return " ".join(selected_traits)

    def _run_inference(self, original, gaps):
        # 1. Python makes the decision using JSON data
        target_identity = self._get_random_attributes(gaps)
        print(f"🎲 Python Selected: {target_identity}")

        # 2. Llama-3 does the writing
        prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a text rewriting engine. 
Your task is to insert the following adjectives into the user's prompt: "{target_identity}".

Rules:
1. Do not add names.
2. Do not add background details.
3. Keep the sentence structure simple.
4. Output ONLY the rewritten prompt.

Input: "{original}" [Insert: {target_identity}]
Output: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        inputs = self.tokenizer(prompt_template, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.1,
            do_sample=True
        )

        full_text = self.tokenizer.decode(outputs[0])
        clean_text = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].replace("<|eot_id|>",
                                                                                                  "").strip()

        if clean_text.startswith('"') and clean_text.endswith('"'):
            clean_text = clean_text[1:-1]

        return {"rewritten": clean_text}

    @modal.fastapi_endpoint(method="POST")
    def rewrite(self, data: dict):
        original = data.get("original_prompt", "")
        gaps = data.get("gaps", [])
        if not original or not gaps: return {"error": "Missing data"}
        return self._run_inference(original, gaps)

    @modal.method()
    def rewrite_dev(self, data: dict):
        original = data.get("original_prompt", "")
        gaps = data.get("gaps", [])
        return self._run_inference(original, gaps)

