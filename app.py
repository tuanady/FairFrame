import streamlit as st
import pandas as pd
import random
import time

# ==========================================
# 1. CONFIGURATION & KNOWLEDGE BASE
# ==========================================

# A Dictionary of "Risky" Roles that often lead to stereotypical outputs
RISK_RULES = {
    "ceo": ["Gender", "Age", "Ethnicity"],
    "leader": ["Gender", "Age"],
    "boss": ["Gender"],
    "doctor": ["Gender"],
    "nurse": ["Gender"],
    "engineer": ["Gender"],
    "teacher": ["Gender"],
    "cleaning": ["Ethnicity", "Class"],
    "housekeeper": ["Gender", "Ethnicity"],
    "inmate": ["Ethnicity"],
    "criminal": ["Ethnicity"],
    "family": ["Structure", "Ethnicity"], 
    "couple": ["Orientation"] 
}

# Attribute Pools for the "Single Person" Cycling Strategy
ATTR_ETHNICITY = ["Black", "Asian", "Hispanic", "White", "Middle Eastern", "Indigenous", "Multi-racial"]
ATTR_GENDER = ["female", "male", "non-binary"]
ATTR_AGE = ["young", "middle-aged", "elderly"]
ATTR_DISABILITY = ["using a wheelchair", "with a hearing aid", "wearing glasses", ""]

# ==========================================
# 2. LOGIC FUNCTIONS
# ==========================================

def detect_bias(prompt):
    """
    Scans the prompt for occupation/role keywords that risk stereotype amplification.
    """
    prompt_lower = prompt.lower()
    detected_risks = []
    
    for role, axes in RISK_RULES.items():
        if role in prompt_lower:
            detected_risks.append({
                "role": role,
                "axes": axes,
                "msg": f"The term '{role}' often defaults to specific {', '.join(axes)} stereotypes."
            })
            
    return detected_risks

def generate_suggestions(prompt, detected_risks):
    """
    Generates two specific types of fairness-enhanced prompts based on the project goal.
    """
    suggestions = {}
    
    # If no risk detected, just return the prompt
    if not detected_risks:
        return {"Original": prompt}

    # 1. GROUP STRATEGY: Explicit diversity cues for multiple people
    suggestions["Group"] = f"A diverse group of {detected_risks[0]['role']}s of different genders and ethnic backgrounds, {prompt.replace(detected_risks[0]['role'], '').strip()}"

    # 2. SINGLE PERSON STRATEGY: Cycling through varied examples
    # We pick random attributes to "break" the default model bias
    e = random.choice(ATTR_ETHNICITY)
    g = random.choice(ATTR_GENDER)
    a = random.choice(ATTR_AGE)
    d = random.choice(ATTR_DISABILITY)
    
    # Construct specific description (e.g., "An Asian man with a disability...")
    attributes = f"{a} {e} {g}"
    if d:
        attributes += f" {d}"
    
    # Replace the risky word with the detailed description
    # e.g. "CEO" -> "Asian male CEO"
    role = detected_risks[0]['role']
    suggestions["Single"] = prompt.lower().replace(role, f"{attributes} {role}").capitalize()
    
    return suggestions

def mock_image_generation(prompt):
    """
    Simulates the image generation by creating a URL that displays the text.
    (Replace this with Stable Diffusion API calls if you have a GPU/API Key)
    """
    time.sleep(1.0) # Simulate processing
    safe_text = prompt.replace(" ", "%20")
    # Generates a placeholder image containing the prompt text
    return f"https://dummyimage.com/600x400/2d2d2d/fff&text={safe_text}"

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="FairFrame Assistant", layout="wide")

# Initialize Session State for Data Collection
if 'evaluation_data' not in st.session_state:
    st.session_state.evaluation_data = []

st.title("⚖️ FairFrame: Bias-Aware Prompt Assistant")
st.markdown("""
**Goal:** Detect stereotype risks in prompts and suggest fairer versions to make images more diverse and inclusive.
**Status:** Running locally.
""")
st.divider()

# --- INPUT SECTION ---
col_input, col_output = st.columns([1, 2])

with col_input:
    st.subheader("1. Input Prompt")
    user_prompt = st.text_area("Enter prompt here:", "CEO giving a presentation", height=100)
    
    if st.button("Analyze & Suggest", type="primary"):
        st.session_state.analyzed = True
        st.session_state.current_prompt = user_prompt

# --- OUTPUT SECTION ---
with col_output:
    if st.session_state.get('analyzed'):
        st.subheader("2. Detection & Suggestions")
        
        # A. Detection Report
        risks = detect_bias(st.session_state.current_prompt)
        if risks:
            st.error(f"⚠️ **Stereotype Risk Detected**")
            for r in risks:
                st.write(f"• **{r['role'].capitalize()}**: Defaults to {', '.join(r['axes'])}")
        else:
            st.success("✅ No high-risk terms detected.")

        # B. Suggestions
        st.markdown("### 💡 Recommended Enhancements")
        
        suggestions = generate_suggestions(st.session_state.current_prompt, risks)
        
        # Create Tabs for the different strategies
        tab_orig, tab_group, tab_single = st.tabs(["Original (Biased)", "Group (Diverse)", "Single (Specific)"])
        
        # -- Tab: Original --
        with tab_orig:
            st.caption(f"Prompt: {st.session_state.current_prompt}")
            st.image(mock_image_generation(st.session_state.current_prompt), caption="Baseline Output")
            st.warning("Likely Output: High stereotype amplification (e.g., White Male default).")

        # -- Tab: Group --
        with tab_group:
            p_group = suggestions.get("Group", st.session_state.current_prompt)
            st.info(f"**Prompt:** {p_group}")
            st.image(mock_image_generation(p_group), caption="Group Strategy Output")
            
        # -- Tab: Single --
        with tab_single:
            p_single = suggestions.get("Single", st.session_state.current_prompt)
            st.info(f"**Prompt:** {p_single}")
            st.image(mock_image_generation(p_single), caption="Single Person Strategy Output")
            if st.button("🔄 Cycle Attribute (Try Again)"):
                # This triggers a re-run, effectively 'cycling' the random attributes
                st.rerun()

        # ==========================================
        # 4. EVALUATION / DATA COLLECTION
        # ==========================================
        st.divider()
        st.subheader("📝 Evaluation (Data Collection)")
        st.write("Rate the 'Single Person' image above for your project metrics.")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            realism = st.slider("Realism Score (1-5)", 1, 5, 3)
        with c2:
            fairness = st.slider("Fairness/Diversity (1-5)", 1, 5, 3)
        with c3:
            saved = st.button("Save Result to Log")
            
        if saved:
            # Log the data for the report
            entry = {
                "Original_Prompt": st.session_state.current_prompt,
                "Enhanced_Prompt": p_single,
                "Realism": realism,
                "Fairness": fairness,
                "Timestamp": time.strftime("%H:%M:%S")
            }
            st.session_state.evaluation_data.append(entry)
            st.success("Data Point Saved!")

# --- DISPLAY LOGS ---
if st.session_state.evaluation_data:
    st.divider()
    st.subheader("📊 Collected Experiment Data")
    df = pd.DataFrame(st.session_state.evaluation_data)
    st.dataframe(df)