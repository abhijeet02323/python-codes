import streamlit as st

# Function to simulate chatbot responses
def get_bot_response(user_input):
    # For simplicity, this function just echoes the user input
    # You can integrate with a real chatbot model here
    return f"You said: {user_input}"

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f0f0;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput>div>input {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        color: #333;
    }
    .stTextArea>div>textarea {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        color: #333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stTitle {
        font-family: 'Arial', sans-serif;
        color: #4CAF50;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app layout
st.markdown('<p class="stTitle">Chatbot with Streamlit</p>', unsafe_allow_html=True)

# Input from the user
user_input = st.text_input("You:", "")

# Display chatbot response when user input is provided
if user_input:
    response = get_bot_response(user_input)
    st.text_area("Bot:", response, height=100)

