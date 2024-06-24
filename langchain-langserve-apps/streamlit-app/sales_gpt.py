from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

import streamlit as st

load_dotenv()

template = """
You're a manager of a team of sales people. You help your employees in their development, and will take action to ensure 
that they succeed. Use the following context to help answer questions about succeeding in the sales industry. 

{context}

Please cite your sources. If the question is not related to sales, business, or leadership, please say that the question is outside
the scope.

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name="sales",
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10},
)

rag_chain = (
        RunnableParallel(
            {"context": retriever,
             "question": RunnablePassthrough(),
             "temperature": RunnablePassthrough()
             })
        | prompt
        | ChatOpenAI(name="gpt-4")
        | StrOutputParser()
)

chain = rag_chain

st.title('Sales-GPT')


import streamlit as st
import pandas as pd
from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)


with st.form('my_form'):
    text = st.text_area('Enter sales-related question')
    submitted = st.form_submit_button('Submit')
    if submitted:
        response = st.write_stream(rag_chain.stream(text))
