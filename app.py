import streamlit as st
import time
from src.generator import FewShotGenerator

# Page Config with a cute title icon
st.set_page_config(page_title="Sym Train AI", page_icon="🤖", layout="wide")

# Custom CSS for a "Cute & Clean" look
st.markdown("""
<style>
    /* Button Styling */
    .stButton>button {
        background-color: #FF4B4B;
        color: oranged;
        border-radius: 20px;
        padding: 10px 24px;
        border: none;
        font-weight: bold;
    }
    /* Card Styling */
    .category-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #f0f2f6;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .category-card h1 {
        font-size: 50px;
        margin-bottom: 0px;
    }
    /* Input Area Styling */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the logic engine
@st.cache_resource
def get_generator():
    return FewShotGenerator()

# --- HEADER SECTION ---
st.title("🤖 Sym Train Simulation Intelligence")
st.markdown("##### *Your friendly automated customer support assistant*")
st.divider()

# --- 3 CUTE CATEGORY CARDS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="category-card">
        <h1>💳</h1>
        <h3>Payments</h3>
        <p>Updates & Issues</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="category-card">
        <h1>🚗</h1>
        <h3>Insurance</h3>
        <p>Claims & Accidents</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="category-card">
        <h1>📦</h1>
        <h3>Orders</h3>
        <p>Tracking & Status</p>
    </div>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.header(" Test Zone")
    st.info("Click the copy icon to try a scenario:")
    
    with st.expander(" Payment Scenarios", expanded=True):
        st.code("Hi, I ordered a shirt last week and paid with my American Express card. I need to update the payment method because there is an issue with that card. Can you help me?", language=None)
        st.code("Hi, I need to update the payment method for one of my recent orders. Can you help me with that?", language=None)

    with st.expander(" Insurance Scenarios", expanded=True):
        st.code("Hi, I am Sam. I was in a car accident this morning and need to file an insurance claim. Can you help me?", language=None)
        st.code("Hi, can you help me file a claim?", language=None)
        
    with st.expander(" Order Status Scenarios", expanded=True):
        st.code("Hi, I recently ordered a book online. Can you give me an update on the order status?", language=None)
        st.code("Hi, I have been waiting for two weeks for the book I ordered. What is going on with it? Can you give me an update?", language=None)

# --- MAIN INPUT ---
st.subheader("How can I help you today?")

# Load models with a spinner
try:
    with st.spinner("Wakeing up the AI Brain... "):
        generator = get_generator()
    st.caption(" Local Models Ready")
except Exception as e:
    st.error(f"Error loading models: {e}")

user_input = st.text_area("Paste a customer transcript here:", height=100)

if st.button(" Generate Plan ", type="primary"):
    if not user_input:
        st.warning("Please tell me what the customer wants first!")
    else:
        with st.spinner("Reading transcript & thinking..."):
            # Call the generator
            result = generator.generate_steps(user_input)
            
            # --- RESULTS SECTION ---
            st.markdown("---")
            col_res_1, col_res_2 = st.columns([2, 1])
            
            with col_res_1:
                st.subheader(f" Category: {result.get('category', 'Unknown')}")
                st.info(f"**Reasoning:** {result.get('reason', 'N/A')}")
                
                st.write("### Suggested Steps")
                steps = result.get('steps', [])
                if steps:
                    for i, step in enumerate(steps):
                        st.markdown(f"**{i+1}.** {step}")
                else:
                    st.write("No steps found.")

            with col_res_2:
                st.subheader(" System JSON")
                st.caption("Backend Output Format")
                st.json(result)

st.markdown("---")
st.caption("Sym Train Project | 100% Local Privacy Preserving AI")