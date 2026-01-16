import streamlit as st
import pandas as pd
import requests
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Classifier Model
ROUTER_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

# Generator Model (Switched to Mistral-7B for better reliability)
GEN_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"

API_TOKEN = "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"

ZERO_SHOT_LABELS = [
    "This text reinforces a gender role stereotype",
    "This text contains racial or ethnic bias",
    "This text marginalizes people based on age",
    "This text assumes ability or excludes disability",
    "This text is neutral and inclusive"
]

LABEL_EXPLANATIONS = {
    "This text reinforces a gender role stereotype":
    "Associates roles, traits, or behaviors with a specific gender in a way that reinforces traditional or limiting stereotypes.",
    "This text contains racial or ethnic bias":
    "Reflects assumptions, exclusions, or preferences based on race or ethnicity, either explicitly or implicitly.",
    "This text marginalizes people based on age":
    "Favors or excludes individuals due to age, reinforcing age-related stereotypes or invisibility.",
    "This text assumes ability or excludes disability":
    "Assumes all individuals are able-bodied or cognitively typical, or excludes representation of people with disabilities.",
    "This text is neutral and inclusive":
    "Does not privilege or marginalize any group and represents people in a balanced, inclusive manner."
}

RISK_RULES = {
    "ceo": "In 2025, women of color held 7% of C-suite roles; Black women held only 0.4% of Fortune 500 CEO positions.",
    "nurse": "Global nursing remains ~88% female, reinforcing a caregiver gender stereotype.",
    "software engineer": "Women represent ~25% of developers; Hispanic/Latinx developers represent ~8% in major tech hubs.",
    "construction worker": "Women make up only 11% of the industry; disability representation is below 3%.",
    "doctor": "While medical school graduates reached parity, only 5.7% of U.S. physicians identify as Black/African American."
}

# ==========================================
# 2. API HANDLER (FIXED)
# ==========================================

def query_router(payload):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload["task"] = "zero-shot-classification"
    
    # Retry logic for classifier
    for _ in range(3):
        response = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            time.sleep(2)
            continue
        else:
            return {"error": f"HTTP {response.status_code}", "raw": response.text}
    return {"error": "Timeout", "raw": "Model loading too long"}

def query_generator(payload):
    """
    Robust handler for Text Generation.
    Handles Model Loading (503) and API errors.
    """
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload["options"] = {"wait_for_model": True, "use_cache": False}
    
    # Retry up to 5 times if model is loading
    for attempt in range(5):
        try:
            response = requests.post(GEN_URL, headers=headers, json=payload, timeout=45)
            
            if response.status_code == 200:
                return response.json()
            
            # If model is loading (503), wait and retry
            if response.status_code == 503:
                data = response.json()
                wait_time = data.get("estimated_time", 5.0)
                time.sleep(wait_time)
                continue
            
            # If other error, break
            break
            
        except requests.exceptions.RequestException:
            time.sleep(2)
            continue
            
    return None

# ==========================================
# 3. NORMALIZE HF OUTPUT
# ==========================================

def normalize_zero_shot(result):
    if not result: return None
    if isinstance(result, list):
        labels = [item["label"] for item in result if "label" in item]
        scores = [item["score"] for item in result if "score" in item]
    elif isinstance(result, dict) and "labels" in result and "scores" in result:
        labels = result["labels"]
        scores = result["scores"]
    else:
        return None

    if not labels or len(labels) != len(scores):
        return None

    return {"labels": labels, "scores": scores}

# ==========================================
# 4. BIAS ANALYSIS
# ==========================================

def analyze_bias_hybrid(prompt, threshold):
    risks = []
    prompt_lower = prompt.lower()
    
    for role, stat in RISK_RULES.items():
        if role in prompt_lower:
            risks.append({
                "type": "Historical Stat",
                "label": f"{role.title()} Representation",
                "msg": stat,
                "score": 1.0
            })

    payload = {
        "inputs": prompt,
        "parameters": {"candidate_labels": ZERO_SHOT_LABELS, "multi_label": False}
    }

    raw_result = query_router(payload)
    normalized = normalize_zero_shot(raw_result)

    if not normalized:
        return risks, {"error": "Invalid zero-shot output", "raw": raw_result}

    score_map = dict(zip(normalized["labels"], normalized["scores"]))

    for label, score in score_map.items():
        if "neutral" not in label.lower() and score >= threshold:
            risks.append({
                "type": "AI Semantic",
                "label": label.replace("This text ", "").capitalize(),
                "msg": f"Detected with confidence {int(score * 100)}% (≥ {int(threshold * 100)}%).",
                "score": score
            })

    return risks, normalized

# ==========================================
# 5. SCORE ANALYSIS HELPER
# ==========================================

def analyze_prompt_scores(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {"candidate_labels": ZERO_SHOT_LABELS, "multi_label": False}
    }
    raw_result = query_router(payload)
    normalized = normalize_zero_shot(raw_result)
    
    if not normalized: return None

    return pd.DataFrame({
        "Label": normalized["labels"],
        "Score": normalized["scores"]
    }).sort_values("Score", ascending=False)

def get_inclusion_score(prompt):
    """Helper to get just the Neutral/Inclusive score."""
    if not prompt: return 0.0
    
    payload = {
        "inputs": prompt,
        "parameters": {"candidate_labels": ZERO_SHOT_LABELS, "multi_label": False}
    }
    raw = query_router(payload)
    norm = normalize_zero_shot(raw)
    if not norm: return 0.0
    
    score_map = dict(zip(norm["labels"], norm["scores"]))
    return score_map.get("This text is neutral and inclusive", 0.0)

# ==========================================
# 6. FAIR PROMPT GENERATOR (AUTOMATED LOOP)
# ==========================================

def generate_fair_prompt_llm(original_prompt, risks):
    """
    Generates recommendations naturally using an LLM.
    Verifies that 'Neutral/Inclusive' score is >= 0.60.

    Improvements:
    - Do not mutate instruction across retries.
    - Explicitly forbid repeating the original prompt.
    - Post-process to remove echoes of the original prompt and provide a descriptive fallback.
    """
    
    detected_issues = [r['label'] for r in risks if r['type'] == 'AI Semantic']
    issue_str = ", ".join(detected_issues) if detected_issues else "potential stereotypes"

    # Using Mistral Instruction format with stricter wording
    strategies = {
        "single": (
            f"You are a DEI expert. The user prompt '{original_prompt}' may contain {issue_str}. "
            f"Write exactly ONE high-quality image generation prompt for a specific individual that counters this by depicting "
            f"an underrepresented demographic (specific ethnicity, age, or visible disability) in a dignified, professional way. "
            f"Do NOT repeat or quote the user's original prompt. Do NOT include any intro or meta-text. Output must be a single clean image prompt only."
        ),
        "group": (
            f"You are a DEI expert. Rewrite the prompt to depict a highly diverse GROUP of people (mix of genders, specific distinct ethnicities, "
            f"various ages, and visible disabilities) collaborating in a dignified scene. "
            f"Do NOT reuse or repeat the user's original prompt text ('{original_prompt}'). Do NOT include intros or commentary—only the final image prompt."
        )
    }

    final_outputs = {}

    for mode, instr_base in strategies.items():
        best_text = ""
        best_score = -1.0
        
        # Try up to 4 times to find a high-scoring prompt
        for i in range(4):
            # Build a fresh instruction each attempt (do not mutate instr_base)
            stricter_suffix = ""
            if i > 0:
                stricter_suffix = " Emphasize concrete demographic details and avoid generalities. Use respectful, specific descriptors."
            instruction = instr_base + stricter_suffix

            # Formulate prompt for Mistral
            full_prompt = f"<s>[INST] {instruction} [/INST]"

            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_new_tokens": 120,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            
            gen_res = query_generator(payload)
            if not gen_res:
                continue

            # Support both list response and single-dict response shapes
            candidate = ""
            if isinstance(gen_res, list) and len(gen_res) > 0:
                candidate = gen_res[0].get("generated_text", "")
            elif isinstance(gen_res, dict):
                candidate = gen_res.get("generated_text", "") or gen_res.get("text", "")
            candidate = (candidate or "").strip()

            # Cleanup obvious artifacts and remove any quoted original prompt
            candidate = candidate.replace('"', '').replace("Here is a prompt:", "").strip()
            if original_prompt and original_prompt.strip() in candidate:
                candidate = candidate.replace(original_prompt.strip(), "").strip(" -:;,.\"'")

            # If the model still mostly echoed the input or result is too short, skip
            if len(candidate) < 12:
                continue

            # Verify Score
            score = get_inclusion_score(candidate)
            
            if score > best_score:
                best_score = score
                best_text = candidate
            
            # STRICT THRESHOLD CHECK
            if score >= 0.60:
                break 

        # If API failed completely or still echoed input, provide explicit fallback that is clearly distinct
        if not best_text or original_prompt.strip() in best_text:
            # Build an explicit fallback prompt for group vs single
            if mode == "group":
                best_text = (
                    "Wide-angle photo of a diverse team in a modern office: two women (one Black, one East Asian), "
                    "one non-binary person (Latinx) using a wheelchair, an older South Asian man, and a younger white woman "
                    "collaborating around a laptop; natural lighting, professional attire, candid interaction, respectful depiction, "
                    "high detail, 50mm portrait lens."
                )
            else:
                best_text = (
                    "Portrait of a mid-career Latinx woman doctor with a visible limb difference, standing confidently in a clinic, "
                    "soft natural light, professional attire, dignified expression, high detail, 85mm lens."
                )

        final_outputs[mode] = best_text

    return final_outputs

# ==========================================
# 7. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="⚖️ FairPrompt", layout="wide")
st.title("⚖️ FairPrompt — Explainable Bias Detection")

if "raw_ai" not in st.session_state:
    st.session_state.raw_ai = None
    st.session_state.risks = []

with st.sidebar:
    st.header("Settings")
    bias_threshold = st.slider("Bias Threshold", 0.05, 1.0, 0.22)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Input Prompt")
        user_input = st.text_input("Prompt:", "A portrait of a hardworking cleaner")

        if st.button("Run Analysis", type="primary"):
            risks, res = analyze_bias_hybrid(user_input, bias_threshold)
            st.session_state.raw_ai = res
            st.session_state.risks = risks

    with col_right:
        res = st.session_state.raw_ai
        risks = st.session_state.risks

        if res:
            if "error" in res:
                st.error("❌ Model Output Error")
                st.json(res)
            else:
                st.subheader("2. AI Zero-Shot Score Distribution")
                df = pd.DataFrame({
                    "Label": res["labels"],
                    "Score": res["scores"]
                }).sort_values("Score", ascending=False)
                st.table(df.reset_index(drop=True))

st.divider()
st.subheader("3. Findings & Recommendations")

if st.session_state.raw_ai:
    if st.session_state.risks:
        for r in st.session_state.risks:
            st.warning(f"⚠️ **{r['label']}**")
            st.write(r["msg"])
    else:
        st.success("✅ No bias category crossed the threshold.")

    # GENERATION & VERIFICATION
    st.markdown("---")
    with st.spinner("🤖 AI is rewriting prompts and verifying scores > 0.60... (This may take a moment)"):
        alts = generate_fair_prompt_llm(user_input, st.session_state.risks)

    t1, t2 = st.tabs(["👤 Inclusive Individual", "👥 Inclusive Group"])

    with t1:
        st.info(alts["single"])
        ind_df = analyze_prompt_scores(alts["single"])
        if ind_df is not None:
            n_score = ind_df[ind_df['Label'] == "This text is neutral and inclusive"]['Score'].values[0]
            # Color logic based on threshold
            color = "green" if n_score >= 0.60 else "red"
            st.markdown(f"**Neutrality Score:** :{color}[{n_score:.3f}] (Target: ≥ 0.60)")
            with st.expander("View Score Details"):
                st.table(ind_df.reset_index(drop=True))

    with t2:
        st.info(alts["group"])
        grp_df = analyze_prompt_scores(alts["group"])
        if grp_df is not None:
            n_score = grp_df[grp_df['Label'] == "This text is neutral and inclusive"]['Score'].values[0]
            # Color logic based on threshold
            color = "green" if n_score >= 0.60 else "red"
            st.markdown(f"**Neutrality Score:** :{color}[{n_score:.3f}] (Target: ≥ 0.60)")
            with st.expander("View Score Details"):
                st.table(grp_df.reset_index(drop=True))

else:
    st.info("Run analysis to see findings.")

st.divider()
st.subheader("4. Bias Label Explanations")

for label, explanation in LABEL_EXPLANATIONS.items():
    st.markdown(f"**{label.replace('This text ', '').capitalize()}**")
    st.write(explanation)