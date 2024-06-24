from enum import Enum
from textwrap import dedent
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


class Prompts(Enum):
    rag = dedent(
        """
        Answer the question based only on the following context. Cite the sources, page numbers, and urls of the information you used.
        
        Context: {context}
        
        Question: {question}
        """
    )


def get_pinecone_retriever(index_name, embedding, k):
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        (
            f"Source: {doc.metadata['file_name'].replace('-', ' ').replace('_', ' ')}\nURL: {doc.metadata['url']}"
            f"\nPage Number: {int(doc.metadata['page_number'])}\nContent: {doc.page_content if doc.metadata['category'] != 'Table' else doc.metadata['text_as_html']}"
        )
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)


gpt_4o = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)
prompt_template = ChatPromptTemplate.from_template(Prompts.rag.value)
retriever = get_pinecone_retriever(
    index_name="lp-hr-rag",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    k=5,
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_template
    | gpt_4o
    | StrOutputParser()
)

st.title("LP HR")

with st.chat_message("user"):
    st.write("Hello 👋")


with st.form("my_form"):

    text = st.text_area(
        "Enter text:",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        response = st.write_stream(rag_chain.stream(text))
