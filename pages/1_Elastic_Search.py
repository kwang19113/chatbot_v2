import streamlit as st
import time
import numpy as np
import os ,sys
from langchain.prompts import ChatPromptTemplate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prompt_layout
from General import generate_query,generate_response,get_par



st.set_page_config(page_title="⛃Elastic Search⛃", page_icon="⛃")
st.title("🤖OTS ASSISTANT")

st.markdown("Get information from server⛃")
if 'messages' not in st.session_state:
        st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message['role'],).markdown(message['content'])
prompt = st.chat_input("Ask me a question")
if prompt:
    #display user message
    st.chat_message('user').markdown(prompt)
    #store message
    st.session_state.messages.append({'role':'user','content':prompt})  
    try:
        response = get_par(prompt,)
        st.chat_message('assistant').markdown(response)
        #store message
        st.session_state.messages.append({'role':'assistant','content':response})  

    except Exception as e:
            print(f"Đã có lỗi xảy ra: {e}")