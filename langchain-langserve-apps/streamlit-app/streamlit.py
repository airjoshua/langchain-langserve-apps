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
You're a manager of a team of sales people. Write a helpful, inspiring e-mail to a sales person that will help them achieve their goals, based on the following context:

{context}

Please cite your sources.

Question: {question}
"""


prompt = ChatPromptTemplate.from_template(template)


llm = ChatOpenAI(name="gpt-4")

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
             "question": RunnablePassthrough()
             })
        | prompt
        | llm
        | StrOutputParser()
)




chain = rag_chain

st.title('Sales-GPT')

with st.form('my_form'):
    text = st.text_area()
    submitted = st.form_submit_button('Submit')
    st.info(rag_chain.invoke(text))
