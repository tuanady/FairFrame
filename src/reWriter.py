# import modal
# import random
# import json
# import re
# from pathlib import Path
#
# # 1. SETUP PATHS
# script_path = Path(__file__).resolve()
# project_root = script_path.parent.parent
#
# # 2. SETUP IMAGE
# image = (
#     modal.Image.debian_slim()
#     .pip_install(
#         "torch",
#         "transformers",
#         "accelerate",
#         "bitsandbytes",
#         "fastapi[standard]"
#     )
#     .add_local_file(project_root / "demographics.json", "/root/demographics.json")
# )
#
# app = modal.App("fairframe-rewriter")
#
#
# @app.cls(image=image, gpu="L4", timeout=600, container_idle_timeout=300)
# class Rewriter:
#
#     @modal.enter()
#     def setup(self):
#         from transformers import AutoTokenizer, AutoModelForCausalLM
#
#         # 1. Load Demographics
#         print("⬇️ Loading Demographics JSON...")
#         self.demographics = {}
#         try:
#             with open("/root/demographics.json", "r") as f:
#                 self.demographics = json.load(f)
#             print(f"✅ Loaded Options: {list(self.demographics.keys())}")
#         except Exception as e:
#             print(f"⚠️ WARNING: Could not load demographics.json: {e}")
#
#         # 2. Load Llama Model
#         print("⬇️ Loading Llama-3 for rewriting...")
#         model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit"
#
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 device_map="auto",
#                 trust_remote_code=True
#             )
#             print("✅ Llama Rewriter loaded!")
#         except Exception as e:
#             print(f"❌ MODEL CRASH: {e}")
#             raise e
#
#     def _extract_profession(self, prompt):
#         """
#         Extract the profession/role from the prompt to ensure it's preserved
#         """
#         # Common STEM and general professions
#         professions = [
#             'researcher', 'scientist', 'engineer', 'biologist', 'chemist',
#             'physicist', 'mathematician', 'developer', 'programmer', 'architect',
#             'doctor', 'surgeon', 'physician', 'dentist', 'veterinarian',
#             'professor', 'teacher', 'analyst', 'technician', 'designer',
#             'astronaut', 'geologist', 'botanist', 'zoologist', 'geneticist',
#             'pharmacist', 'nurse', 'pilot', 'ceo', 'manager', 'director',
#             'neurologist', 'cardiologist', 'radiologist', 'pathologist',
#             'software engineer', 'data scientist', 'machine learning engineer',
#             'robotics engineer', 'aerospace engineer', 'mechanical engineer',
#             'electrical engineer', 'civil engineer', 'chemical engineer',
#             'environmental scientist', 'marine biologist', 'astrophysicist',
#             'quantum physicist', 'microbiologist', 'biochemist', 'ecologist',
#             'meteorologist', 'oceanographer', 'paleontologist', 'virologist',
#             'immunologist', 'epidemiologist', 'anesthesiologist', 'pediatrician'
#         ]
#
#         prompt_lower = prompt.lower()
#
#         # Check for multi-word professions first (longer matches first)
#         professions_sorted = sorted(professions, key=len, reverse=True)
#         for prof in professions_sorted:
#             if prof in prompt_lower:
#                 return prof
#
#         return None
#
#     def _get_random_attributes(self, gaps):
#         """
#         Picks random traits ONLY for gaps with score > 60%
#         Input: ["Race", "Age", "Disability"]
#         Output: "Black, elderly, in a wheelchair"
#         """
#         selected_traits = []
#
#         if not self.demographics:
#             # Fallback if JSON missing
#             if "Race" in gaps: selected_traits.append("Black")
#             if "Gender" in gaps: selected_traits.append("female")
#             if "Age" in gaps: selected_traits.append("middle-aged")
#             if "Disability" in gaps: selected_traits.append("in a wheelchair")
#             return ", ".join(selected_traits)
#
#         for gap in gaps:
#             options = self.demographics.get(gap, [])
#             if options:
#                 trait = random.choice(options)
#                 selected_traits.append(trait)
#
#         return ", ".join(selected_traits)
#
#     def _run_inference(self, original_prompt, gaps):
#         """
#         Rewrite the prompt to include diversity attributes
#         """
#         # Only process if there are gaps
#         if not gaps:
#             return {"rewritten": original_prompt, "note": "No gaps detected"}
#
#         target_identity = self._get_random_attributes(gaps)
#
#         if not target_identity:
#             return {"rewritten": original_prompt, "note": "No attributes selected"}
#
#         # Extract profession to emphasize preservation
#         profession = self._extract_profession(original_prompt)
#         profession_reminder = ""
#         if profession:
#             profession_reminder = f"\n\n**CRITICAL**: The profession is '{profession}' - you MUST keep this EXACT word in your rewrite!"
#
#         # Construct rewriting prompt
#         prompt_template = f"""<|start_header_id|>system<|end_header_id|>
# You are an expert AI editor optimizing image prompts for diversity.
#
# Your Goal: Rewrite the "User Prompt" to include the "New Attributes" while PRESERVING any diversity adjectives already in the text.
#
#
# *IDENTIFY SUBJECT & NUMBER:**
# - check if the main subject is **Singular** (e.g., "doctor", "pilot") or **Plural** (e.g., "team", "group", "engineers").
# - If **Singular**: Apply attributes directly to the person.
# - If **Plural**: Apply attributes to the *members* of the group (e.g., "A team of [Adjectives] engineers").
#
# **CRITICAL INSTRUCTIONS:**
# - **Identify the Profession:** Find the main role (e.g., "doctor", "researcher"). Do NOT change this word.
# - **Merge Attributes:** Combine traits already in the prompt (e.g., "Black") with the "New Attributes" provided.
#
# **APPLY THE GOLDEN ORDER:**
# Construct the sentence in this order for maximum clarity:
#
# **[Prefix] + [AGE] + [RACE] + [GENDER] + [PROFESSION/GROUP] + [DISABILITY/VISUAL TRAITS] + [Setting/Action]**
#
# *Important for Disabilities:* Attributes starting with "in", "with", or "using" (e.g., "in a wheelchair", "with vitiligo") MUST go **AFTER** the profession.
#
# **Examples:**
#
# * **Input:** "A Black doctor"
#     **New Attributes:** "female, young, in a wheelchair"
#     **Logic:** Merge "Black" (existing) + "female, young, in a wheel chair" (new).
#     **Output:** "A young Black female doctor"
#
# * **Input:** "An old researcher in a lab"
#     **New Attributes:** "Asian, in a wheelchair, male"
#     **Logic:** Order: Age (old) -> Race (Asian) -> Gender (male) -> Prof (researcher) -> Disability (in a wheelchair).
#     **Output:** "An old Asian male researcher in a wheelchair in a lab"
#
# * **Input:** "Generate a black female software engineer coding"
#     **New Attributes:** " deaf, middle-aged"
#     **Logic:** Order: Age (middle-aged) -> Race (black) -> gender (female) -> Prof (software engineer) -> Disability (who is deaf).
#     **Output:** "Generate a middle-aged Black female software engineer who is deaf coding"
#
# * **Input:** "An aerospace team in a lab"
#   **Attributes:** "Middle Eastern, queer, elderly, with a cochlear implant"
#   **Logic:** Subject is "team" (Plural). "Queer" describes the members. "With a cochlear implant" goes after the noun.
#   **Output:** "An aerospace team of elderly Middle Eastern queer engineers with cochlear implants in a lab"
#
# **Task:**
# User Prompt: "{original_prompt}"
# New Attributes: "{target_identity}"
#
# Output ONLY the rewritten prompt. No quotes.
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
#
#         inputs = self.tokenizer(prompt_template, return_tensors="pt").to("cuda")
#
#         outputs = self.model.generate(
#             inputs.input_ids,
#             max_new_tokens=100,
#             temperature=0.2,  # Lower temperature for more consistent output
#             do_sample=True,
#             pad_token_id=self.tokenizer.eos_token_id
#         )
#
#         full_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
#         clean_text = full_text.strip()
#
#         # Remove quotes if Llama adds them
#         if clean_text.startswith('"') and clean_text.endswith('"'):
#             clean_text = clean_text[1:-1]
#
#         # VALIDATE: Check if profession is preserved
#         if profession and profession not in clean_text.lower():
#             print(f"⚠️ WARNING: Profession '{profession}' was lost in rewrite!")
#             print(f"   Original had: '{profession}'")
#             print(f"   Rewrite has: '{clean_text}'")
#
#             # Try to fix by inserting profession
#             # Look for generic words to replace
#             for generic in ['person', 'individual', 'professional', 'worker']:
#                 if generic in clean_text.lower():
#                     clean_text = re.sub(
#                         rf'\b{generic}\b',
#                         profession,
#                         clean_text,
#                         count=1,
#                         flags=re.IGNORECASE
#                     )
#                     print(f"   ✅ Fixed: Replaced '{generic}' with '{profession}'")
#                     break
#
#         print(f"📝 Original: '{original_prompt}'")
#         print(f"🎯 Gaps: {gaps}")
#         print(f"🌈 Attributes: {target_identity}")
#         if profession:
#             print(f"💼 Profession: {profession}")
#         print(f"✨ Rewritten: '{clean_text}'")
#
#         return {
#             "rewritten": clean_text,
#             "attributes_added": target_identity,
#             "gaps_addressed": gaps,
#             "profession_preserved": profession
#         }
#
#     @modal.web_endpoint(method="POST")
#     def rewrite(self, data: dict):
#         """
#         Web endpoint for rewriting prompts
#
#         Input:
#         {
#           "original_prompt": "a researcher in a lab",
#           "gaps": ["Race", "Gender", "Disability"]  // Only gaps with >60% score
#         }
#
#         Output:
#         {
#           "rewritten": "a Black female researcher in a wheelchair in a lab",
#           "attributes_added": "Black, female, in a wheelchair",
#           "gaps_addressed": ["Race", "Gender", "Disability"],
#           "profession_preserved": "researcher"
#         }
#         """
#         original = data.get("original_prompt", "")
#         gaps = data.get("gaps", [])
#
#         print(f"🔄 REWRITER INPUT RECEIVED:")
#         print(f"   - Prompt: '{original}'")
#         print(f"   - Gaps:   {gaps}")
#
#         if not original:
#             return {"error": "Missing original_prompt"}
#
#         if not gaps:
#             return {
#                 "rewritten": original,
#                 "note": "No gaps to address - prompt is already inclusive"
#             }
#
#         return self._run_inference(original, gaps)
#
#
# # Test the rewriter
# @app.function(image=image, gpu="L4")
# def test_rewriter():
#     """Test the rewriter with sample inputs"""
#     rewriter = Rewriter()
#
#     test_cases = [
#         {
#             "original_prompt": "a researcher in a lab",
#             "gaps": ["Race", "Gender", "Disability"]
#         },
#         {
#             "original_prompt": "a female biologist",
#             "gaps": ["Race", "Age", "Disability"]
#         },
#         {
#             "original_prompt": "Generate a software engineer",
#             "gaps": ["Gender", "Race", "Age", "Disability"]
#         },
#         {
#             "original_prompt": "A young Black woman scientist",
#             "gaps": ["Disability"]
#         },
#         {
#             "original_prompt": "A biologist with a prosthetic arm using a microscope",
#             "gaps": ["Gender", "Race", "Age"]
#         },
#         {
#             "original_prompt": "An architect reviewing blueprints",
#             "gaps": ["Gender", "Race", "Age", "Disability"]
#         },
#         {
#             "original_prompt": "A chemist mixing solutions in a beaker",
#             "gaps": ["Gender", "Race", "Age", "Disability"]
#         },
#         {
#             "original_prompt": "Give me a picture of a data scientist analyzing code",
#             "gaps": ["Gender", "Race", "Age", "Disability"]
#         }
#     ]
#
#     print("\n" + "=" * 60)
#     print("🎨 TESTING REWRITER - PROFESSION PRESERVATION")
#     print("=" * 60 + "\n")
#
#     for case in test_cases:
#         result = rewriter._run_inference(
#             case["original_prompt"],
#             case["gaps"]
#         )
#         print("-" * 60 + "\n")
#
#
# @app.local_entrypoint()
# def main():
#     """Run test"""
#     test_rewriter.remote()
#
#

import modal
import os
import json
import re
from pathlib import Path

# 1. SETUP IMAGE
# Added "fastapi[standard]" to fix the web endpoint error
image = (
    modal.Image.debian_slim()
    .pip_install("google-genai", "fastapi[standard]")
    .add_local_file("demographics.json", "/root/demographics.json")
)

app = modal.App("fairframe-rewriter")


# 2. DEFINE THE CLASS
@app.cls(
    image=image,
    # ⚠️ Keep your API key secure. Ideally, use modal.Secret.
    secrets=[modal.Secret.from_dict({"GOOGLE_API_KEY": "AIzaSyB5pTnE1VYJY-ejGC1NCsa66o8vyokFAzM"})]
)
class Rewriter:
    @modal.enter()
    def setup(self):
        from google import genai

        print("⬇️ Loading Gemini Client...")
        # The SDK automatically picks up the key from the environment variable set by secrets
        self.client = genai.Client()
        self.demographics = self._load_demographics()
        print("✅ Gemini Rewriter Ready!")

    def _load_demographics(self):
        try:
            with open("/root/demographics.json", "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load demographics: {e}")
            return {}

    def _extract_profession(self, prompt):
        # Extensive list of professions to preserve
        professions = [
            'researcher', 'scientist', 'engineer', 'biologist', 'chemist',
            'physicist', 'mathematician', 'developer', 'programmer', 'architect',
            'doctor', 'surgeon', 'physician', 'dentist', 'veterinarian',
            'professor', 'teacher', 'analyst', 'technician', 'designer',
            'astronaut', 'geologist', 'botanist', 'zoologist', 'geneticist',
            'pharmacist', 'nurse', 'pilot', 'ceo', 'manager', 'director',
            'neurologist', 'cardiologist', 'radiologist', 'pathologist',
            'software engineer', 'data scientist', 'machine learning engineer',
            'robotics engineer', 'aerospace engineer', 'mechanical engineer',
            'electrical engineer', 'civil engineer', 'chemical engineer',
            'environmental scientist', 'marine biologist', 'astrophysicist',
            'quantum physicist', 'microbiologist', 'biochemist', 'ecologist',
            'meteorologist', 'oceanographer', 'paleontologist', 'virologist',
            'immunologist', 'epidemiologist', 'anesthesiologist', 'pediatrician',
            'team', 'group', 'staff', 'crew'
        ]

        prompt_lower = prompt.lower()
        # Sort by length to match specific jobs first
        professions_sorted = sorted(professions, key=len, reverse=True)

        for prof in professions_sorted:
            # Use regex to match whole words only
            if re.search(rf"\b{re.escape(prof)}\b", prompt_lower):
                return prof
        return None

    def _get_random_attributes(self, gaps):
        selected_traits = []
        if not self.demographics:
            # Fallback defaults if JSON is missing
            defaults = {"Race": "Black", "Gender": "female", "Age": "middle-aged", "Disability": "in a wheelchair"}
            for gap in gaps:
                if gap in defaults: selected_traits.append(defaults[gap])
            return ", ".join(selected_traits)

        for gap in gaps:
            options = self.demographics.get(gap, [])
            if options:
                import random
                trait = random.choice(options)
                selected_traits.append(trait)

        return ", ".join(selected_traits)

    def _run_inference(self, original_prompt, gaps):
        if not gaps:
            return {"rewritten": original_prompt, "note": "No gaps detected"}

        target_identity = self._get_random_attributes(gaps)
        if not target_identity:
            return {"rewritten": original_prompt, "note": "No attributes selected"}

        profession = self._extract_profession(original_prompt)

        # System Prompt construction
        system_instruction = """You are an expert AI editor optimizing image prompts for diversity.
Your Goal: Rewrite the "User Prompt" to include the "New Attributes" naturally.

**CRITICAL RULES:**
1. **IDENTIFY SUBJECT & NUMBER:**
   - If Singular: Apply adjectives to the person.
   - If Plural (team, group): Apply adjectives to the members.
2. **PRESERVE THE PROFESSION:** Do NOT change the main role.
3. **APPLY THE GOLDEN ORDER:**
   [Prefix] + [AGE] + [RACE] + [GENDER] + [PROFESSION] + [DISABILITY/VISUAL TRAITS] + [Setting/Action]
   *Note:* Disability traits starting with "in", "with", or "using" must go AFTER the profession.
"""

        user_content = f"""
User Prompt: "{original_prompt}"
New Attributes: "{target_identity}"
Important: The profession/subject is likely "{profession if profession else 'unknown'}". Keep it.

Output ONLY the rewritten prompt text. Do not add quotes.
"""

        try:
            # Call Gemini 2.0 Flash
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=system_instruction + user_content
            )

            clean_text = response.text.strip()
            if clean_text.startswith('"') and clean_text.endswith('"'):
                clean_text = clean_text[1:-1]

            # Validation: Restore profession if lost
            if profession and profession.lower() not in clean_text.lower():
                print(f"⚠️ Fixing lost profession: {profession}")
                for generic in ['person', 'individual', 'professional', 'worker']:
                    if generic in clean_text.lower():
                        clean_text = re.sub(rf'\b{generic}\b', profession, clean_text, flags=re.IGNORECASE)
                        break

            print(f"✅ Rewritten: {clean_text}")

            return {
                "rewritten": clean_text,
                "attributes_added": target_identity,
                "gaps_addressed": gaps,
                "profession_preserved": profession
            }

        except Exception as e:
            print(f"❌ Gemini Error: {e}")
            return {"error": str(e), "rewritten": original_prompt}

    # 3. DEFINE WEB ENDPOINT
    @modal.web_endpoint(method="POST")
    def rewrite(self, data: dict):
        original = data.get("original_prompt", "")
        gaps = data.get("gaps", [])

        print(f"📥 Input: {original} | Gaps: {gaps}")
        return self._run_inference(original, gaps)