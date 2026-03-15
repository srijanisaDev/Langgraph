import streamlit as st
from graph import bmi_graph

# --- Page Config ---
st.set_page_config(page_title="BMI Analyzer", page_icon="🏋️", layout="centered")

st.title("🏋️ BMI Analyzer")
st.caption("Powered by LangGraph + Gemini AI")
st.divider()

# --- Input Form ---
with st.form("bmi_form"):
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.5)
    with col2:
        height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.75, step=0.01)
    
    submitted = st.form_submit_button("🔍 Analyze", use_container_width=True)

# --- Run Graph & Display Results ---
if submitted:
    with st.spinner("Running LangGraph workflow..."):
        result = bmi_graph.invoke({
            "weight": weight,
            "height": height,
            "bmi": 0.0,
            "category": "",
            "advice": ""
        })

    st.divider()

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="BMI Score", value=result['bmi'])
    with col2:
        st.metric(label="Category", value=result['category'])

    # Color-coded category badge
    category = result['category']
    color_map = {
        "Underweight": "🔵",
        "Normal": "🟢",
        "Overweight": "🟠",
        "Obese": "🔴"
    }
    st.markdown(f"### {color_map.get(category, '')} {category}")

    # AI Advice
    st.info(f"💡 **AI Advice:** {result['advice']}")
