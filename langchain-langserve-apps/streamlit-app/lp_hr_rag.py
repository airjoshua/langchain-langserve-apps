from enum import Enum
from textwrap import dedent
from time import time
from typing import Iterable, List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import ChatCohere, CohereRerank
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableGenerator,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import streamlit as st

load_dotenv()


class Prompts(Enum):
    rag = dedent(
        """
        {llama_3}You are an assistant that can answer questions about, and explain, LP's documents. Cite your sources, including
        the 1) 'file name', 2) 'page number', and 3) 'url' that informed your answer. If you don't how to respond, just say that you don't know.
        ############
        LP's documents: {context}
        ############
        Question: {question}
        ############
        Answer:
        """
    )


class LlamaPrompts(Enum):
    rag = dedent(
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant that can answer questions about, and explain, LP's documents. Cite your sources, including
        the 1) 'file name', 2) 'page number', and 3) 'url' that informed your answer. If you don't how to respond, 
        just say that you don't know. <|eot_id|><|start_header_id|>user<|end_header_id|>
        ############
        Question: {question} 
        ############
        LP's documents: {context}
        ############ 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    )
    grader = dedent(
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    )
    hallucination_grader = dedent(
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>`
    """
    )
    answer_grader = dedent(
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    )


def get_pinecone_retriever(index_name, embedding, k):
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def format_docs(docs: List[Document]) -> str:
    formatted = (
        (
            f"File Name: {doc.metadata['file_name'].replace('-', ' ').replace('_', ' ')}\nURL: {doc.metadata['url']}"
            f"\nPage Number: {int(doc.metadata['page_number'])}\nContent: {doc.page_content if doc.metadata['category'] != 'Table' else doc.metadata['text_as_html']}"
        )
        for doc in docs
    )

    return "\n\n" + "\n\n".join(formatted)


def parse(ai_message) -> str:
    """Parse the AI message."""
    return ai_message.content.swapcase()


def streaming_parse(chunks: Iterable[AIMessage]) -> Iterable[str]:
    for chunk in chunks:
        yield chunk.content.swapcase()


def get_rag_chain_from_docs():
    retriever = get_pinecone_retriever(
        index_name="lp",
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        k=5,
    )
    gpt_4o = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)
    # streaming_parse = RunnableGenerator(streaming_parse)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | ChatPromptTemplate.from_template(Prompts.rag.value)
        | gpt_4o
        | StrOutputParser()
    )

    retrieve_docs = (lambda x: x["question"]) | retriever

    # rag_chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
    #     answer=rag_chain_from_docs
    # )

    return RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)


def get_basic_rag_chain(model):
    gpt_4o = init_chat_model(
        model="gpt-4o",
        model_provider="openai",
        temperature=0,
    )
    llama_3 = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
    )
    cohere = ChatCohere(model="command-r")
    models = {"cohere": cohere, "GPT-4o": gpt_4o, "LLama 3": llama_3}
    retriever = get_pinecone_retriever(
        index_name="lp",
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        k=5,
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=CohereRerank(),
        base_retriever=retriever,
    )
    prompt = ChatPromptTemplate.from_template(
        LlamaPrompts.rag.value if "llama_3" else Prompts.rag.value
    )

    return (
        {
            "context": compression_retriever | format_docs,  # compression_retriever
            "question": RunnablePassthrough(),
        }
        | prompt
        | models[model]
        | StrOutputParser()
    )


def test_llm(chain):
    questions = [
        "What are the gift limits?",
        "Give a thorough explanation of our dental benefits",
        "Describe the medical plans.",
        "What are the deductibles for the blue plan?",
        "What services are not covered in the white plan?",
        "Explain the different health plans available to salaried employees",
        "Describe our dental benefits for salaried employees",
        "what days are holidays?",
        "Explain our personal leave policy",
        "What is the hybrid work policy?",
        "What's the out of pocket limit for the blue plan?",
        "What is the deductible for the Orange Plan?",
        "Can I relocate while still at LP?",
        "I've witnessed bad behavior by a co-worker, but I am afraid of speaking out due to fear of retaliation. What are my protections?",
    ]
    for question in questions:
        print(question)
        print("\n")
        print(chain.invoke(question))
        print("################")


def query_pc(query, index, embeddings, k=3):
    vector_store = PineconeVectorStore(index=index, embeddings=embeddings)
    return vector_store.similarity_search(
        query=query,
        k=k,
    )


#rag_chain = get_basic_rag_chain(model="llama_3")
#test_llm(rag_chain)

#print(rag_chain.invoke("What is the hybrid work policy?"))

# response = rag_chain.invoke("What is the hybrid work policy?")


st.title("LP")

with st.chat_message("user"):
    st.write("Ask about ")

model_selection = st.selectbox(
    "Pick an LLM",
    ("GPT-4o", "LLama 3"))
rag_chain = get_basic_rag_chain(model=model_selection)
st.write("You chose:", model_selection)
with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        response = st.write_stream(rag_chain.stream(text))
