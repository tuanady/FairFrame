import streamlit as st
import pandas as pd
import random
import datetime
import requests
import time

# ==========================================
# 1. CONFIGURATION & UPDATED KNOWLEDGE BASE
# ==========================================
ROUTER_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
API_TOKEN = "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" 

# Level 1: Static Rules (Updated with 2026 Diversity Stats)
RISK_RULES = {
    "ceo": "In 2025, women of color held 7% of C-suite roles; Black women held only 0.4% of Fortune 500 CEO spots.",
    "nurse": "Global nursing remains ~88% female, reinforcing a caregiver gender stereotype.",
    "software engineer": "Women represent ~25% of developers; Hispanic/Latinx devs represent only ~8% in major tech hubs.",
    "construction worker": "Women make up only 11% of the industry; visible disability representation is below 3%.",
    "doctor": "While medical school graduates have reached parity, only 5.7% of US physicians identify as Black/African American."
}

# Granular attributes for Example Generation
ATTR_ETHNICITY = ["Indigenous", "South Asian", "Black", "Latinx", "East Asian", "Middle Eastern", "Afro-Latino"]
ATTR_GENDER = ["non-binary person", "woman", "man", "gender-fluid person"]
ATTR_AGE = ["an early-career (20s)", "a mid-career (40s)", "a senior (65+)", "an experienced (50s)"]
ATTR_DISABILITY = ["using a wheelchair", "with a prosthetic limb", "with a white cane", "using sign language", ""]

# ==========================================
# 2. CORE LOGIC & API HANDLER
# ==========================================

def query_router(payload):
    """Handles API calls with automatic retry for 503 (Loading) errors."""
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    for attempt in range(5):  # Increased retries for cold starts
        response = requests.post(ROUTER_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            # Model is loading - wait and retry
            estimated_time = response.json().get("estimated_time", 10)
            time.sleep(min(estimated_time, 10))
            continue
        else:
            return {"error": f"API Error {response.status_code}", "details": response.text}
    return None

def analyze_bias_hybrid(prompt):
    prompt_lower = prompt.lower()
    risks = []

    # Layer 1: Statistical Check
    for role, stat in RISK_RULES.items():
        if role in prompt_lower:
            risks.append({"type": "Historical Stat", "label": f"{role.title()} Representation", "msg": stat, "score": 1.0})

    # Layer 2: AI Semantic Analysis
    payload = {
        "inputs": prompt,
        "parameters": {
            "candidate_labels": ["gender role stereotype", "racial bias", "ageist exclusion", "neutral and inclusive"],
            "multi_label": False
        },
    }
    
    result = query_router(payload)
    if result and "labels" in result:
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        if top_label != "neutral and inclusive" and top_score > 0.15:
            risks.append({
                "type": "AI Semantic",
                "label": top_label.replace(" ", " ").title(),
                "msg": f"AI detected bias patterns with {int(top_score*100)}% confidence.",
                "score": top_score,
                "raw": result # Keep for debugging
            })
            
    return risks, result

def generate_fair_prompt(prompt):
    e, g, a, d = random.choice(ATTR_ETHNICITY), random.choice(ATTR_GENDER), random.choice(ATTR_AGE), random.choice(ATTR_DISABILITY)
    return {
        "single": f"A realistic, high-quality professional photo of {a} {e} {g} {f' {d}' if d else ''} as a {prompt.lower()}.",
        "group": f"A diverse group of {prompt.lower()}s of different genders, ages, and ethnicities working together in a modern inclusive setting."
    }

# ==========================================
# 3. STREAMLIT UI & DEBUGGER
# ==========================================
st.set_page_config(page_title="API Debugger & Bias Detector", page_icon="⚖️", layout="wide")

st.title("⚖️ FairPrompt Hybrid: API Test Suite")

# Sidebar Status Monitor
with st.sidebar:
    st.header("🔌 API Connection")
    if st.button("Ping Inference Server"):
        with st.spinner("Testing..."):
            test_res = query_router({"inputs": "health check", "parameters": {"candidate_labels": ["test"]}})
            if test_res and "labels" in test_res:
                st.success("Status: Online")
            else:
                st.error("Status: Offline / Error")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Input Prompt")
    user_input = st.text_input("Enter a role or description:", "A software engineer")
    analyze_btn = st.button("Run Analysis", type="primary")

with col_right:
    if analyze_btn:
        with st.spinner("Analyzing..."):
            risks, raw_ai = analyze_bias_hybrid(user_input)
            
            # --- THE "HOW I KNOW IT WORKS" SECTION ---
            st.subheader("2. API Integrity Check")
            if "error" in raw_ai:
                st.error(f"❌ API failure: {raw_ai['error']}")
            else:
                st.success("✅ API returned data successfully.")
                with st.expander("🔍 View Raw JSON from Hugging Face"):
                    st.json(raw_ai)
            
            st.divider()
            
            # --- RESULTS SECTION ---
            if risks:
                for r in risks:
                    st.warning(f"⚠️ **{r['label']}** ({r['type']})")
                    st.write(r['msg'])
                
                alts = generate_fair_prompt(user_input)
                t1, t2 = st.tabs(["Specific Individual", "Diverse Group"])
                with t1: st.info(alts['single'])
                with t2: st.info(alts['group'])
            else:
                st.success("No significant biases detected.")