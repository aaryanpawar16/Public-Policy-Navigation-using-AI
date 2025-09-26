import streamlit as st                          #NAME-K.SATYA SAMPATH KUMAR MAIL-ZYWU801@GMAIL.COM
file = st.file_uploader("Choose a file", type=["txt", "pdf", "csv"])
if "history" not in st.session_state:
    st.session_state.history = []
msg = st.chat_input("Say something...")
if msg:
    st.session_state.history.append(msg)
for m in st.session_state.history:
    st.write("You:", m)
