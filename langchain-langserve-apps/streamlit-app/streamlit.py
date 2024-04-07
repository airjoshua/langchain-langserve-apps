import streamlit as st
from sales_chatbot.sales_chat import rag_chain

st.title('Sales-GPT')

with st.form('my_form'):
    text = st.text_area('Enter text:', "What would you like to talk about?")
    submitted = st.form_submit_button('Submit')
    st.info(rag_chain.invoke(text))
