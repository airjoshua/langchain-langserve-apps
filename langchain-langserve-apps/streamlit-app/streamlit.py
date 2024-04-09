import streamlit as st
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

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

with st.form('my_form'):
    text = st.text_area('Enter sales-related question')
    submitted = st.form_submit_button('Submit')
    if submitted:
        response = st.write_stream(rag_chain.stream(text))
