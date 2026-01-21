import modal
import json
import re



app = modal.App("STEMagine-classifier")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "fastapi[standard]"
    )
)

@app.cls(image=image, gpu="L4", timeout=600, container_idle_timeout=300)
class LlamaClassifier:

    @modal.enter()
    def setup(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("⬇️ Loading Llama 3 8B Instruct...")
        model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Llama Classifier loaded!")

    # @modal.method()
    def classify(self, prompt_text):
        """
        Classify diversity gaps using ONLY a prompt - no training needed!
        """

        # THE MAGIC: This prompt is ALL you need!
        system_prompt = """You are a diversity gap analyzer for image generation prompts.

Your task: Identify which demographic dimensions are MISSING (not explicitly mentioned) in the image prompt.

**Dimensions to check (EACH dimension must be EXPLICITLY stated):**
1. **Gender**: female, male, woman, man, non-binary, transgender, girl, boy, etc.
   - "researcher" = NO gender → GAP
   - "female researcher" = gender present → NOT a gap
   
2. **Race**: Black, Asian, Latino/Latina, Indigenous, White, Middle Eastern, Pacific Islander, African, Hispanic, etc.
   - "researcher" = NO race → GAP
   - "Black researcher" = race present → NOT a gap
   
3. **Age**: young, elderly, senior, middle-aged, mature, old, teen, child, aged, etc.
   - "researcher" = NO age → GAP
   - "young researcher" = age present → NOT a gap
   
4. **Disability**: wheelchair, prosthetic arm/leg/limb, deaf, blind, low vision, hearing aids, autism, ADHD, mobility aid, white cane, sign language, cochlear implant, cerebral palsy, dyslexia, neurodivergent, etc.
   - "researcher" = NO disability → GAP
   - "researcher with a prosthetic arm" = disability present → NOT a gap

**CRITICAL RULES:**
- EACH dimension must be EXPLICITLY mentioned to NOT be a gap
- "A biologist" has ZERO demographics specified → ALL 4 gaps
- "A biologist with a prosthetic arm" has ONLY disability → Gender, Race, Age gaps remain
- "A Black female biologist" has gender + race → Age, Disability gaps remain
- Generic nouns like "person", "researcher", "engineer" WITHOUT descriptors = ALL gaps
- DO NOT CONFUSE RACE WITH NATIONALITY/ETHNICITY (German is not a race)

**PRESERVE EXISTING ADJECTIVES** 
If the user already specified a trait (e.g., "Black", "Young"), you MUST keep it. Never remove existing descriptors.

**Logic:**
- If the user does NOT explicitly say the age (e.g., "young", "old", "30s"), it is a GAP.
- If the user does NOT explicitly say the gender (e.g., "man", "woman", "female"), it is a GAP.
- If the user does NOT explicitly say the race (e.g., "Black", "Asian", "White"), it is a GAP.
- If the user does NOT explicitly say the disability status, it is a GAP.

**Examples:**
Prompt: "a doctor"
Analysis: NO gender, NO race, NO age, NO disability
Result: {"gaps": ["Gender", "Race", "Age", "Disability"]}

Prompt: "an Asian engineer in a wheelchair"
Analysis: NO gender, YES race (Asian), NO age, YES disability (wheelchair)
Result: {"gaps": ["Gender", "Age"]}

Prompt: "a young Black female scientist with hearing aids"
Analysis: YES gender (female), YES race (Black), YES age (young), YES disability (hearing aids)
Result: {"gaps": []}

**Response format (MUST be valid JSON):**
{
  "gaps": ["Race", "Age", "Disability"],
  "scores": {
    "Gender": 0.05,
    "Race": 0.92,
    "Age": 0.88,
    "Disability": 0.90
  },
  "safe_score": 0.05
}

Rules for scores:
- Missing dimension (gap): 0.85-0.95
- Specified dimension: 0.05-0.15
- safe_score: 0.05-0.10 if gaps exist, 0.90-0.95 if no gaps"""

        # Format as Llama 3 chat
        prompt_template = f"""<|start_header_id|>system<|end_header_id|>
{system_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Analyze this prompt: {prompt_text}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        inputs = self.tokenizer(prompt_template, return_tensors="pt").to("cuda")

        # Generate response
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=200,
            temperature=0.1,  # Low temperature = more deterministic
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        print(f"📝 Raw LLM response: {response}")

        # Parse JSON from response
        return self._parse_response(response)

    def _parse_response(self, response):
        """Extract JSON from LLM response"""
        try:
            # Try to extract JSON block
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)

                # Validate structure
                if "gaps" in result and "scores" in result:
                    return result
        except Exception as e:
            print(f"⚠️ JSON parse error: {e}")

        # Fallback: conservative guess (assume all gaps)
        return {
            "gaps": ["Gender", "Race", "Age", "Disability"],
            "scores": {
                "Gender": 0.85,
                "Race": 0.85,
                "Age": 0.85,
                "Disability": 0.85
            },
            "safe_score": 0.05,
            "parse_error": True
        }

    @modal.web_endpoint(method="POST")
    def analyze(self, data: dict):
        """
        Web endpoint - drop-in replacement for your DeBERTa classifier
        Returns gaps with scores > 60%
        """
        prompt = data.get("prompt", "")

        if not prompt:
            return {"error": "Missing prompt"}

        print(f"🔍 Analyzing: '{prompt}'")
        result = self.classify(prompt)

        # Filter gaps: only include those with score > 0.60 (60%)
        filtered_gaps = []
        for gap in result.get("gaps", []):
            score = result.get("scores", {}).get(gap, 0)
            if score > 0.60:
                filtered_gaps.append(gap)

        # Update result with filtered gaps
        result["gaps"] = filtered_gaps

        print(f"✅ Detected gaps (>60%): {filtered_gaps}")
        result["original_prompt"] = prompt

        return result


# Test function
@app.function(image=image, gpu="L4")
def run_tests():
    """Test the classifier with sample prompts"""
    classifier = LlamaClassifier()

    tests = [
        # "Give me a picture of a Researcher",
        # "Give me a picture of STEM student",
        # "Generate a math professor",
        # "Generate a black female researcher",
        # "Generate a picture of a male chemist",
         "Give me a picture of a white engineer",
        # "A female biologist in a lab",
        "give a picture of an asian engineer in a wheel chair",
        "A biologist with a prosthetic arm using a microscope",
    ]

    print("\n" + "=" * 60)
    print("🔬 LLAMA GAP ANALYSIS (>60% threshold)")
    print("=" * 60 + "\n")

    for text in tests:
        result = classifier.classify.remote(text)

        # Filter to >60%
        filtered_gaps = []
        for gap in result.get("gaps", []):
            score = result.get("scores", {}).get(gap, 0)
            if score > 0.60:
                filtered_gaps.append((gap, score))

        filtered_gaps.sort(key=lambda x: x[1], reverse=True)

        if not filtered_gaps:
            final_tag = "✅ Safe"
        else:
            tag_names = [x[0] for x in filtered_gaps]
            final_tag = "/".join(tag_names) + " Gap"

        print(f"📝 Prompt: '{text}'")
        print(f"   -> Detected Tag:    {final_tag}")

        if filtered_gaps:
            details = ", ".join([f"{n} ({s:.1%})" for n, s in filtered_gaps])
            print(f"   -> Scores:          {details}")
        print("-" * 60)


@app.local_entrypoint()
def main():
    """Run test"""
    run_tests.remote()