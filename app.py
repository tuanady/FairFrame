import streamlit as st
import pandas as pd
import random
import datetime
import requests
import time

# ==========================================
# 1. HYBRID CONFIGURATION & KNOWLEDGE BASE
# ==========================================
# 2026 Router Endpoint
ROUTER_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
# Using the provided token
API_TOKEN = "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" 

# Level 1: Static Rules (Guaranteed to catch specific biased terms)
# This ensures that even if the AI is 'unsure', the research stats will flag it
RISK_RULES = {
    "ceo": "In 2025, women of color hold less than 7% of C-suite roles in major tech firms.",
    "nurse": "Global nursing remains ~88% female, contributing to a severe gender-based caregiving stereotype.",
    "software engineer": "Women represent only ~25% of the global dev workforce as of 2025.",
    "construction worker": "Women make up only 11% of construction roles; disability is rarely represented."
}

ATTR_ETHNICITY = ["Indigenous", "South Asian", "Black", "Latinx", "East Asian", "Middle Eastern", "Mixed-race"]
ATTR_GENDER = ["non-binary", "woman", "man", "gender-fluid"]
ATTR_AGE = ["early-career", "middle-aged", "senior (60+)"]
ATTR_DISABILITY = ["using a wheelchair", "with a prosthetic limb", "with a white cane", "with a hearing aid", ""]

# ==========================================
# 2. LOGIC FUNCTIONS
# ==========================================

def query_router(payload):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    for attempt in range(3):
        try:
            response = requests.post(ROUTER_URL, headers=headers, json=payload, timeout=30)
            if response.status_code in [503, 429]:
                wait = response.json().get("estimated_time", 10)
                time.sleep(min(wait, 10))
                continue
            response.raise_for_status()
            return response.json()
        except:
            continue
    return None

def analyze_bias_hybrid(prompt):
    """Hybrid Detection: Dictionary check + High-Sensitivity AI."""
    prompt_lower = prompt.lower()
    risks = []

    # Level 1: Static Rule Check (Dictionary-based)
    # This addresses the 'everything is neutral' bug for specific roles
    for role, stat in RISK_RULES.items():
        if role in prompt_lower:
            risks.append({
                "type": "Historical Stat", 
                "label": f"{role.title()} Stereotype", 
                "msg": stat,
                "score": 1.0 # High confidence for rule-based match
            })

    # Level 2: Semantic AI Detection (Low threshold to catch subtle bias)
    payload = {
        "inputs": prompt,
        "parameters": {
            "candidate_labels": [
                "stereotypical gender roles", 
                "racial or ethnic stereotypes", 
                "occupational bias", 
                "age-based exclusion", 
                "neutral and inclusive"
            ],
            "multi_label": False
        },
    }
    
    result = query_router(payload)
    if result and "labels" in result:
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        # LOWER THRESHOLD: 0.18 is used to catch subtle semantic patterns
        if top_label != "neutral and inclusive" and top_score > 0.18:
            risks.append({
                "type": "AI Semantic",
                "label": top_label.title(),
                "msg": f"AI Confidence Score: {int(top_score*100)}%",
                "score": top_score
            })
            
    return risks

def generate_fair_prompt(prompt):
    e, g, a, d = random.choice(ATTR_ETHNICITY), random.choice(ATTR_GENDER), random.choice(ATTR_AGE), random.choice(ATTR_DISABILITY)
    diversity_context = f"{a} {e} {g}" + (f" {d}" if d else "")
    return {
        "single": f"A realistic, high-quality photo of a {diversity_context} {prompt.lower()}.",
        "group": f"A diverse group of people from varied backgrounds and genders, {prompt.lower()}."
    }

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="FairPrompt Hybrid", page_icon="⚖️", layout="wide")

if 'logs' not in st.session_state:
    st.session_state.logs = []

st.title("⚖️ FairPrompt Hybrid: Bias-Aware Assistant")
st.markdown("### Combined Statistical (Static) & Semantic (AI) Analysis")

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("1. Enter Original Prompt")
    user_input = st.text_area("What do you want to generate?", "A nurse in a hospital", height=100)
    analyze_btn = st.button("Run Hybrid Analysis", type="primary", use_container_width=True)

with col_right:
    if analyze_btn and user_input:
        with st.spinner("Analyzing semantics via Router..."):
            risks = analyze_bias_hybrid(user_input)
            
            if risks:
                for r in risks:
                    st.error(f"⚠️ **[{r['type']}] {r['label']}**")
                    st.write(r['msg'])
                
                st.divider()
                st.subheader("2. Suggested Inclusive Versions")
                alts = generate_fair_prompt(user_input)
                
                # Tabbed results for research clarity
                tab1, tab2 = st.tabs(["Specific Individual", "Diverse Group"])
                with tab1:
                    st.success(alts['single'])
                    st.caption("Strategy: Counter-Stereotype Attribute Injection")
                with tab2:
                    st.success(alts['group'])
                    st.caption("Strategy: Collective Diversity Parity")
                
                if st.button("💾 Log Result to Dataset"):
                    st.session_state.logs.append({
                        "Timestamp": datetime.datetime.now().strftime("%H:%M"),
                        "Input": user_input,
                        "Bias_Type": risks[0]['label'],
                        "Confidence": risks[0]['score'],
                        "Enhanced": alts['single']
                    })
                    st.toast("Saved!")
            else:
                st.success("✅ No major stereotypes detected in this prompt.")

# --- RESEARCH DATASET ---
if st.session_state.logs:
    st.divider()
    st.subheader("📊 Collected Experiment Data")
    df = pd.DataFrame(st.session_state.logs)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), "fairprompt_research_data.csv")