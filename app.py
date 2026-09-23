import os
import streamlit as st
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="Placements - Ready AI Agent 6465",
    page_icon="🎓",
    layout="wide"
)

# Application Title & Details
st.title("🎓 Placements - Ready AI Agent 6465")
st.subheader("Career Coach AI for Student Interview & Placement Prep")

# Sidebar - Configuration
st.sidebar.header("Agent Settings")
api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")

model_choice = st.sidebar.selectbox(
    "Select Model:",
    ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
)

# Instructions in Sidebar
st.sidebar.markdown("""
### Agent Details
- **Agent Name:** playground-v1 / Placements - Ready AI Agent 6465
- **Target Users:** Placement cells & Students
- **Focus:** HR questions, coding prep, aptitude simulation
""")

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query Input
user_input = st.chat_input("Ex: Ask me HR questions for TCS...")

if user_input:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar to proceed.")
    else:
        # Append User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate Response using OpenAI
        client = OpenAI(api_key=api_key)
        
        system_prompt = (
            "You are a Placement Readiness AI Agent. "
            "Help students prepare for campus placements, coding, aptitude, and HR interviews. "
            "When asked for HR questions or practice, provide structured questions, "
            "evaluation criteria, sample responses, and a Placement Readiness Score out of 100."
        )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing query and generating interview prep..."):
                try:
                    response = client.chat.completions.create(
                        model=model_choice,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            *st.session_state.messages
                        ],
                        temperature=0.7
                    )
                    
                    bot_reply = response.choices[0].message.content
                    st.markdown(bot_reply)
                    
                    # Store Assistant Response
                    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
