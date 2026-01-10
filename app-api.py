import streamlit as st
import pandas as pd
import random
import requests

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROUTER_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
API_TOKEN = "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"

ZERO_SHOT_LABELS = [
    "gender role stereotype",
    "racial bias",
    "ageist exclusion",
    "neutral and inclusive"
]

RISK_RULES = {
    "ceo": "In 2025, women of color held 7% of C-suite roles; Black women held only 0.4% of Fortune 500 CEO spots.",
    "nurse": "Global nursing remains ~88% female, reinforcing a caregiver gender stereotype.",
    "software engineer": "Women represent ~25% of developers; Hispanic/Latinx devs represent only ~8% in major tech hubs.",
    "construction worker": "Women make up only 11% of the industry; visible disability representation is below 3%.",
    "doctor": "While medical school graduates have reached parity, only 5.7% of US physicians identify as Black/African American."
}

ATTR_ETHNICITY = ["Indigenous", "South Asian", "Black", "Latinx", "East Asian", "Middle Eastern", "Afro-Latino"]
ATTR_GENDER = ["non-binary person", "woman", "man", "gender-fluid person"]
ATTR_AGE = ["an early-career (20s)", "a mid-career (40s)", "a senior (65+)", "an experienced (50s)"]
ATTR_DISABILITY = ["using a wheelchair", "with a prosthetic limb", "with a white cane", "using sign language", ""]

# ==========================================
# 2. API HANDLER
# ==========================================

def query_router(payload):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload["task"] = "zero-shot-classification"
    payload["options"] = {"wait_for_model": True}

    response = requests.post(
        ROUTER_URL,
        headers=headers,
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        return {"error": f"HTTP {response.status_code}", "raw": response.text}

    return response.json()

# ==========================================
# 3. NORMALIZE HF OUTPUT (ROBUST)
# ==========================================

def normalize_zero_shot(result):
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
# 4. BIAS ANALYSIS (FIXED THRESHOLD LOGIC)
# ==========================================

def analyze_bias_hybrid(prompt, threshold):
    risks = []

    # ---- Historical rule-based layer
    prompt_lower = prompt.lower()
    for role, stat in RISK_RULES.items():
        if role in prompt_lower:
            risks.append({
                "type": "Historical Stat",
                "label": f"{role.title()} Representation",
                "msg": stat,
                "score": 1.0
            })

    # ---- AI semantic layer
    payload = {
        "inputs": prompt,
        "parameters": {
            "candidate_labels": ZERO_SHOT_LABELS,
            "multi_label": False
        }
    }

    raw_result = query_router(payload)
    normalized = normalize_zero_shot(raw_result)

    if not normalized:
        return risks, {"error": "Invalid zero-shot output", "raw": raw_result}

    score_map = dict(zip(normalized["labels"], normalized["scores"]))

    neutral_score = score_map.get("neutral and inclusive", 0.0)

    bias_hits = []

    for bias_label in [
        "gender role stereotype",
        "racial bias",
        "ageist exclusion"
    ]:
        if score_map.get(bias_label, 0.0) >= threshold:
            bias_hits.append({
                "type": "AI Semantic",
                "label": bias_label.title(),
                "msg": (
                    f"{bias_label.title()} detected "
                    f"({int(score_map[bias_label]*100)}% ≥ {int(threshold*100)}%)."
                ),
                "score": score_map[bias_label]
            })

    # ---- FINAL DECISION (YOUR RULES)
    if bias_hits:
        risks.extend(bias_hits)
    # else: neutral dominates OR no strong signal → no bias

    return risks, normalized

# ==========================================
# 5. FAIR PROMPT GENERATOR
# ==========================================

def generate_fair_prompt(prompt):
    e = random.choice(ATTR_ETHNICITY)
    g = random.choice(ATTR_GENDER)
    a = random.choice(ATTR_AGE)
    d = random.choice(ATTR_DISABILITY)

    return {
        "single": (
            f"A realistic, high-quality professional photo of "
            f"{a} {e} {g}{f' {d}' if d else ''} as a {prompt.lower()}."
        ),
        "group": (
            f"A diverse group of {prompt.lower()}s of different genders, "
            f"ages, and ethnicities collaborating in an inclusive environment."
        )
    }

# ==========================================
# 6. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="⚖️ FairPrompt (Final Logic)", layout="wide")
st.title("⚖️ FairPrompt — Threshold-Exact Bias Detection")

if "raw_ai" not in st.session_state:
    st.session_state.raw_ai = None
    st.session_state.risks = []

with st.sidebar:
    st.header("Settings")
    bias_threshold = st.slider("Bias Threshold", 0.05, 1.0, 0.22)

col_left, col_right = st.columns(2)

# ---- INPUT
with col_left:
    st.subheader("1. Input Prompt")
    user_input = st.text_input("Prompt:", "A portrait of a hardworking cleaner")

    if st.button("Run Analysis", type="primary"):
        risks, res = analyze_bias_hybrid(user_input, bias_threshold)
        st.session_state.raw_ai = res
        st.session_state.risks = risks

# ---- OUTPUT
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

            if risks:
                for r in risks:
                    st.warning(f"⚠️ **{r['label']}**")
                    st.write(r["msg"])

                alts = generate_fair_prompt(user_input)
                t1, t2 = st.tabs(["👤 Inclusive Individual", "👥 Inclusive Group"])
                with t1:
                    st.success(alts["single"])
                with t2:
                    st.success(alts["group"])
            else:
                st.success("✅ No bias category crossed the threshold.")
