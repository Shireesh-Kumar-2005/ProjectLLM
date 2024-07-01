import streamlit as st
from LLM import ask 

st.title("Agriculture Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for i in st.session_state.messages:
    with st.chat_message(i["role"]):
        st.markdown(i["content"])
        
prompt = st.chat_input("type here....")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)      
    st.session_state.messages.append({"role":"user","content":prompt})
    sk = ask(prompt)
    with st.chat_message("assistant"):
        st.markdown(sk)
    st.session_state.messages.append({"role":"assistant","content":sk})
