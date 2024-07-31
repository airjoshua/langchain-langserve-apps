import datetime
import os
from enum import Enum
from textwrap import dedent
from time import time
from typing import Iterable, List
from langchain_core.runnables import ConfigurableField

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import ChatCohere, CohereRerank

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    RunnableGenerator,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from arize_otel import register_otel, Endpoints

from openinference.instrumentation.langchain import LangChainInstrumentor
import streamlit as st

load_dotenv()


class Prompts(Enum):
    rag_ = dedent(
        """
        You are an assistant that can answer questions about, and explain, LP's documents. Cite your sources, including
        the 1) 'file name', 2) 'page number', and 3) 'url' that informed your answer. If you don't how to respond, just say that you don't know.
        ############
        Question: {question}
        ############
        LP's documents: {context}
        ############
        Answer:
        """
    )
    rag = (
        # "You are an assistant that can answer questions about, and explain, LP's benefits documents. Cite your sources, "
        # # "including the 1) 'file name', 2) 'page number', and 3) 'url' that informed your answer. If you don't how to "
        # "including the file name, page number, and url that informed your answer. If you don't how to "
        # "respond, just say that you don't know. "
        # "including the 1) 'file name', 2) 'page number', and 3) 'url' that informed your answer. If you don't how to "
        # "including the file name, page number, and url that informed your answer. If you don't how to "
        # "respond, just say that you don't know. "
        "You are an HR benefits assistant that can answer questions about, and explain, LP's benefits documents. "
        "If the question is related to LP's medical plans, please explain differences in coverage based on whether the services "
        "are in-network or out-of-network. Moreover, if not explicitly asked about a particular medical plan, such as "
        "the blue, white, or orange plan, please include information pertaining to all 3 plans. Cite your sources, "
        "including the file name, page number, and url that informed your answer. If the answer is not found in LP's documents, do not make inferences, just say 'I don't know.'."
        "\n\n"
        "Question: {question} \n\n"
        "LP's documents: {context} \n\n"
        "Answer:"
    )
    qa_system_prompt = dedent(
        """ 
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.
        "\n\n"
        "{context}"
    """
    )

    contextualize_q_system_prompt = dedent(
        """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone 
        question which can be understood without the chat history. Do NOT answer the question, just reformulate it if 
        needed and otherwise return it as is."""
    )


class LlamaPrompts(Enum):
    rag = dedent(
        # "<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an assistant that can answer questions about"
        # "and explain, LP's documents. Cite your sources, including the 1) 'file name', 2) 'page number', and 3) 'url'\n"
        # "that informed your answer. If you don't how to respond, just say that you don't know.\n"
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        "You are an HR benefits assistant that can answer questions about, and explain, LP's benefits documents. "
        # "If the question is related to LP's medical plans, please explain differences in coverage depending on whether the services "
        # "are in-network or out-of-network. Moreover, if not explicitly asked about a particular medical plan, such as "
        # "the blue, white, or orange plan, please include information pertaining to all 3 plans."
        "Cite your sources, including the file name, page number, and url that informed your answer. "
        "If the answer is not found in LP's documents, do not make inferences, just say 'I don't know.'."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Question: {question}\n\n"
        "Context: {context} \n\n"
        "Answer: "
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

    qa_system_prompt = dedent(
        """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant that can answer questions about, and explain, LP's documents. 
        Cite your sources, including the 1) 'file name', 2) 'page number', and 3) 'url' that informed your answer. If you don't how to respond, 
        just say that you don't know. <|eot_id|><|start_header_id|>user<|end_header_id|>
        ############
        Request: {request}
        ############
        LP's documents: {context}
        ############
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    )
    contextualize_q_system_prompt = dedent(
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|> Given a chat history and the latest user question which might reference 
        context in the chat history, formulate a standalone question which can be understood without the chat history.
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is. <|eot_id|><|start_header_id|>user<|end_header_id|>
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
            # f"\nPage Number: {int(doc.metadata['page_number'])}\nContent: {doc.page_content}"
        )
        for doc in docs
    )

    return "\n\n" + "\n\n".join(formatted)


def parse(ai_message) -> str:
    """Parse the AI message."""
    return ai_message.content.swapcase()


def few_shot_prompting():
    example_1 = dedent(
        """
        The dental benefits for salaried employees are as follows:
            * The maximum payout per person per year for all services, except for cephalometric radiographs, photographs, diagnostic models, and orthodontia, is $1,500. (Source: Dental Summary (Spanish), Page 2, https://lpcorp.com/)
            * There is a separate lifetime maximum of $1,500 per person for cephalometric radiographs, photographs, diagnostic models, and orthodontia services. (Source: Dental Summary (Spanish), Page 2, https://lpcorp.com/)
            * The Delta Dental plan also offers enhanced dental coverage for members who have diabetes, heart disease, or kidney disease, have suffered a stroke, are currently pregnant, have undergone head and neck cancer radiation, or have had an organ transplant. Those who qualify can receive 100% coverage for related dental care. (Source: 2024 LP Benefits Guide, US Salary updated V3, Page 16, https://lpcorp.com/)

        Additionally, new employees' benefits will begin on the first day of the month following their hire date, and dependents over 26 years old who are physically or mentally disabled, as well as pre-Medicare retirees who meet the age/service requirements, are also eligible for coverage. (Source: Dental Summary (Spanish), Page 2, https://lpcorp.com/)
    """
    )
    example_2 = dedent(
        """
        The services that are not covered in the White Plan include:
        1. Cosmetic surgery
        2. Dental care (Adult)
        3. Dental care (Children)
        4. Infertility treatment
        5. Long-term care
        6. Non-emergency care when traveling outside the U.S.
        7. Prescription Drugs
        8. Private-duty nursing
        9. Routine eye care (Adult)
        10. Routine eye care (Children)
        11. Routine foot care for non-diabetics
        12. Weight loss programs

        Source:
        File Name: 2024 White Plan Summary of Benefits
        Page Number: 5
        url: https://lpcorp.com/
    """
    )
    examples = [
        {
            "input": "Describe our dental benefits for salaried employees",
            "output": example_1,
        },
        {
            "input": "What services are not covered in the white plan?",
            "output": example_2,
        },
    ]
    return examples


def streaming_parse(chunks: Iterable[AIMessage]) -> Iterable[str]:
    for chunk in chunks:
        yield chunk.content.swapcase()


def get_rag_chain_from_docs():
    retriever = get_pinecone_retriever(
        index_name="benefits-rag",
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
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


def rag_with_sources(llm, index_name, llm_temperature):
    compression_retriever = get_compression_retriever(
        k=5,
        index_name=index_name,
        # embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    )
    base_retriever = get_pinecone_retriever(
        index_name=index_name,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
        k=5,
    )
    chat_prompt = ChatPromptTemplate.from_template(
        Prompts.rag.value
    ).configurable_alternatives(
        ConfigurableField(id="prompt"),
        default_key="gpt-4o",
        llama_3=ChatPromptTemplate.from_template(LlamaPrompts.rag.value),
    )
    model = (
        init_chat_model(
            model="gpt-4o",
            model_provider="openai",
        )
        .configurable_fields(
            temperature=ConfigurableField(
                id="llm_temperature",
                name="LLM Temperature",
                description="The temperature of the LLM",
            )
        )
        .configurable_alternatives(
            ConfigurableField(id="llm"),
            default_key="gpt-4o",
            llama_3=ChatGroq(model="llama3-70b-8192"),
        )
    )

    from IPython.core.display import HTML, display

    def parse_sources(ai_message: AIMessage) -> str:
        answer = ai_message.content
        sources = (
            f"File Name: {source.metadata["file_name"]} \n"
            f"Page Number: {int(source.metadata["page_number"])} \n"
            f"Text: {source.metadata["text_as_html"] if source.metadata["category"] == 'Table' else source.page_content}"
            for source in ai_message["context"]
        )
        return f"{answer}\n\nSources:\n\n" + "\n\n".join(sources)
        # for source in ai_message['context']:
        #     sources.append(f"File Name: {source.metadata["file_name"]} \n"
        #      f"Page Number: {source.metadata["page_number"].astype(int)} \n"
        #      f"Text: {display(HTML(source.metadata["text_as_html"])) if source.metadata["category"] == 'Table' else source.page_content}")

        # sources = (source.page_content for source in ai_message['context'] if source["metadata"])

    # from langchain_core.runnables import RunnableGenerator
    # rag_chain_from_docs = (
    #     RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    #     | chat_prompt
    #     | model
    #     | StrOutputParser()
    # ).with_config(configurable=configurable)

    # return RunnableParallel(
    #     {
    #         "context": base_retriever,
    #         "request": RunnablePassthrough(),
    #     }
    # ).assign(answer=rag_chain_from_docs)
    # c = RunnableParallel(
    #     {"context": base_retriever,
    #      "request": RunnablePassthrough()
    #      }
    # ).assign(answer=rag_chain_from_docs)
    from langchain_core.runnables import RunnableGenerator

    # intro = {"context": base_retriever,
    #  "request": RunnablePassthrough()
    #  }
    # streaming_parse = RunnableGenerator(streaming_parse)
    # f = (c | parse_sources).stream("what is the deductible for the white plan?")

    # return rag_chain.with_config(configurable=configurable)


def rag_with_alternatives(llm, index_name, llm_temperature):
    # base_retriever = get_pinecone_retriever(
    #     index_name=index_name,
    #     embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    #     k=5,
    # )
    compression_retriever = get_compression_retriever(
        k=10,
        index_name=index_name,
        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    )
    chat_model = init_chat_model(
        model_provider="openai",
        model="gpt-4o",
        temperature=0,
        configurable_fields="any",
    )
    models = {
        "model_providers": {
            "groq": {
                # "llama_8b_3_1": "llama-3.1-8b-instant",
                # "llama_70b_3_1": "llama-3.1-70b-versatile",
                # "llama_405b_3_1": "llama-3.1-405b-reasoning",
                "llama_8b": "llama3-8b-8192",
                "llama_70b": "llama3-70b-8192",
                "llama_70b_tool_use": "llama3-groq-70b-8192-tool-use-preview",
            },
            "openai": {
                "gpt-4o": "gpt-4o",
            },
            # "together": {
            #     "llama_8b_3_1": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            #     "llama_70b_3_1": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            #     "llama_405b_3_1": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            # },
            "fireworks": {
                "llama_8b_3_1": "accounts/fireworks/models/llama-v3p1-8b-instruct",
                "llama_70b_3_1": "accounts/fireworks/models/llama-v3p1-70b-instruct",
                "llama_405b_3_1": "accounts/fireworks/models/llama-v3p1-405b-instruct",
            },
        }
    }
    model_provider = next(
        model_provider
        for model_provider, model_names in models["model_providers"].items()
        if llm in model_names.keys()
    )
    config = {
        "configurable": {
            "model_provider": model_provider,
            "model": models["model_providers"][model_provider][llm],
            "temperature": llm_temperature,
        }
    }

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}"),
            ("ai", "{answer}"),
        ]
    )
    examples = get_few_shot_examples()

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    print(few_shot_prompt.format())
    final_prompt = ChatPromptTemplate.from_messages(
        [
            few_shot_prompt,
            ("human", LlamaPrompts.rag.value if "llama" in llm else Prompts.rag.value),
        ]
    )
    return (
            {
                "context": compression_retriever | format_docs,  # compression_retriever
                "question": RunnablePassthrough(),
            }
            | final_prompt
            | chat_model.with_config(config=config)
            | StrOutputParser()
    )



def get_few_shot_examples():
    examples = [
        {
            "question": "What will I pay if I have a hospital stay?",
            "answer": "If you have a hospital stay, the costs you will incur depend on whether your provider is in-network or out-of-network. Here are the details for each plan:\n\n### 2024 White Plan:\n- **In-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 20% coinsurance\n  - **Physician/surgeon fees:** 20% coinsurance\n- **Out-of-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 40% coinsurance\n  - **Physician/surgeon fees:** 40% coinsurance\n- **Limitations, Exceptions, & Other Important Information:** Prior Authorization required. Your cost share may increase to 50% if not obtained.\n\n**Source:** 2024 White Plan Summary of Benefits, Page 3, [URL](https://lpcorp.com/)\n\n### 2024 Orange Plan:\n- **In-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 20% coinsurance\n  - **Physician/surgeon fees:** 20% coinsurance\n- **Out-of-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 40% coinsurance\n  - **Physician/surgeon fees:** 40% coinsurance\n- **Limitations, Exceptions, & Other Important Information:** Prior Authorization required. Your cost share may increase to 50% if not obtained.\n\n**Source:** 2024 Orange Plan Summary of Benefits, Page 3, [URL](https://lpcorp.com/)\n\n### 2024 Blue Plan:\n- **In-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 20% coinsurance\n  - **Physician/surgeon fees:** 20% coinsurance\n- **Out-of-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 40% coinsurance\n  - **Physician/surgeon fees:** 40% coinsurance\n- **Limitations, Exceptions, & Other Important Information:** Prior Authorization required. Your cost share may increase to 50% if not obtained.\n\n**Source:** 2024 Blue Plan Summary of Benefits, Page 3, [URL](https://lpcorp.com/)\n\nIn summary, for all plans, you will pay 20% coinsurance for in-network providers and 40% coinsurance for out-of-network providers. Prior authorization is required, and failure to obtain it may increase your cost share to 50%.",
        },
        {
            "question": "What is the prescription drug coverage?",
            "answer": "For prescription drug coverage under LP's medical plans, the details vary depending on the plan and the type of medication. Here's a breakdown of the coverage for each plan:\n\n### 2024 White Plan:\n- **Preferred Generic drugs / Non-Preferred Generic drugs:** Retail 30-Day Supply: $25 co-pay, 90-Day Supply: $25 co-pay, Mail Order: $25 co-pay\n- **Preferred brand drugs:** Retail 30-Day Supply: $50 co-pay, 90-Day Supply: $100 co-pay, Mail Order: $100 co-pay\n- **Non-preferred brand drugs:** Retail 30-Day Supply: $75 co-pay, 90-Day Supply: $150 co-pay, Mail Order: $150 co-pay\n\n### 2024 Orange Plan:\n- **Preferred Generic drugs / Non-Preferred Generic drugs:** 20% coinsurance\n- **Preferred brand drugs:** 20% coinsurance\n- **Non-preferred brand drugs:** 20% coinsurance\n- **Preferred Specialty drugs / Non-Preferred Specialty drugs:** 20% coinsurance\n\n### 2024 Blue Plan:\n- **Preferred Generic drugs / Non-Preferred Generic drugs:** 20% coinsurance\n- **Preferred brand drugs:** 20% coinsurance\n- **Non-preferred brand drugs:** 20% coinsurance\n- **Preferred Specialty drugs / Non-Preferred Specialty drugs:** 20% coinsurance\n\n**Important Notes:**\n- **Retail Supply Limit:** up to a 90-day supply\n- **Mail Order Supply Limit:** up to a 90-day supply\n- **Specialty drugs:** limited to a 30-day supply\n- **Preventive care medication:** covered at 100% from the first day of coverage\n- **Combined medical and Rx out-of-pocket maximum:** once met, Rx cost will be covered at 100%\n- **Prior authorization and step therapy requirements:** may apply\n\n**Sources:**\n- 2024 White Plan Summary of Benefits, Page 2, [URL](https://lpcorp.com/)\n- 2024 Orange Plan Summary of Benefits, Page 2, [URL](https://lpcorp.com/)\n- 2024 Blue Plan Summary of Benefits, Page 2, [URL](https://lpcorp.com/)\n- 2024 LP Benefits Guide, Page 8, [URL](https://lpcorp.com/)",
        },
        {
            "question": "What is the out-of-pocket maximum?",
            "answer": "The out-of-pocket maximum is the most you will have to pay for covered services in a year. Here are the details for each of LP's medical plans:\n\n### 2024 White Plan:\n- **In-Network:** $2,000 per person / $4,000 per family\n- **Out-of-Network:** $4,000 per person / $8,000 per family\n\n**Source:** 2024 White Plan Summary of Benefits, Page 1, [URL](https://lpcorp.com/)\n\n### 2024 Orange Plan:\n- **In-Network:** $4,000 per person\n- **Out-of-Network:** $8,000 per person\n\n**Source:** 2024 Orange Plan Summary of Benefits, Page 1, [URL](https://lpcorp.com/)\n\n### 2024 Blue Plan:\n- **In-Network:** $7,000 per person / $14,000 per family\n- **Out-of-Network:** $14,000 per person / $28,000 per family\n\n**Source:** 2024 Blue Plan Summary of Benefits, Page 1, [URL](https://lpcorp.com/)\n\nPlease note that premiums, balance-billing charges, penalties, and healthcare services not covered by the plan do not count towards the out-of-pocket maximum. Once you reach the out-of-pocket maximum, the insurance company will pay 100% of your eligible medical expenses for the rest of the year.",
        },
        {
            "question": "What's the annual deductible in the orange plan?",
            "answer": "For the Orange Plan, the annual deductible is:\n\n* In-network: $2,000 per person / $3,200 individual in a family / $4,000 family\n* Out-of-network: $4,000 per person / $8,000 family\n\n**Sources:**\n\n* 2024 LP Benefits Guide, Page 9, [https://lpcorp.com/](https://lpcorp.com/)\n* 2024 Orange Plan Summary of Benefits, Page 1, [https://lpcorp.com/](https://lpcorp.com/)",
        },
        {
            "question": "How much are dental fillings?",
            "answer": "For dental fillings under LP's dental plan, the coverage details are as follows:\n\n### In-Network (Delta Dental PPO):\n- **Basic Restorative (fillings, simple extraction, root canals):** Plan pays 80% after the deductible.\n\n### Out-of-Network:\n- **Basic Restorative (fillings, simple extraction, root canals):** Plan pays 80% after the deductible.\n\n### Additional Details:\n- **Annual Deductible:** $50 per individual / $150 per family.\n- **Calendar Year Maximum:** $1,500 per member.\n\nSo, for dental fillings, you will pay 20% of the cost after meeting your annual deductible, whether you go to an in-network or out-of-network provider.\n\n**Sources:**\n- 2024 LP Benefits Guide, Page 16, [URL](https://lpcorp.com/)\n- Dental Summary, Page 1, [URL](https://lpcorp.com/)\n- Dental Summary, Page 2, [URL](https://lpcorp.com/)",
        },
    ]
    return examples


##  | final_prompt


def rag_with_few_shot_prompting(llm, index_name, llm_temperature):
    compression_retriever = get_compression_retriever(
        k=5,
        index_name=index_name,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        # embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
    )
    chat_prompt = ChatPromptTemplate.from_template(
        Prompts.rag.value
    ).configurable_alternatives(
        ConfigurableField(id="prompt"),
        default_key="gpt-4o",
        llama_3=ChatPromptTemplate.from_template(LlamaPrompts.rag.value),
    )
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )
    examples = few_shot_prompting()
    examples = [
        {"input": "2 🦜 2", "output": "4"},
        {"input": "2 🦜 3", "output": "5"},
    ]
    # few_shot_prompt = FewShotChatMessagePromptTemplate(
    #     example_prompt=example_prompt,
    #     examples=examples,
    # )

    model = (
        init_chat_model(
            model="gpt-4o",
            model_provider="openai",
        )
        .configurable_fields(
            temperature=ConfigurableField(
                id="llm_temperature",
                name="LLM Temperature",
                description="The temperature of the LLM",
            )
        )
        .configurable_alternatives(
            ConfigurableField(id="llm"),
            default_key="gpt-4o",
            llama_3=ChatGroq(model="llama3-70b-8192"),
        )
    )
    # rag_chain = (
    #     {
    #         "context": compression_retriever | format_docs,  # compression_retriever
    #         "request": RunnablePassthrough(),
    #     }
    #     # | chat_prompt
    #     | few_shot_prompt
    #     | model
    #     | StrOutputParser()
    # )
    # return rag_chain.with_config(configurable=configurable)


def _get_datetime():
    return datetime.now().strftime("%m/%d/%Y, %H:%M:%S")


def get_compression_retriever(k, index_name, embedding):
    base_retriever = get_pinecone_retriever(
        index_name=index_name,
        embedding=embedding,
        k=k,
    )

    return ContextualCompressionRetriever(
        base_compressor=CohereRerank(top_n=k // 2),
        base_retriever=base_retriever,
    )


def test_llm(chain):
    test_questions = [
        "Give a thorough explanation of our dental benefits",
        "Describe the medical plans.",
        "What are the deductibles for the blue plan?",
        "What services are not covered in the white plan?",
        "Explain the different health plans available to salaried employees",
        "Describe our dental benefits for salaried employees",
        "What services are covered in the dental plan?",
        "what days are holidays?",
        "Can I relocate while still at LP?",
        "What is the deductible for the Orange Plan?",
        "what is the deductible for the white plan?",
        "What's the annual deductible in the blue plan?",
        "what is the out of pocket limit for the blue plan?",
        "what is the out of pocket limit for the white plan?",
        "what is the out of pocket limit for the orange plan?",
        "What's the annual deductible in the blue plan?",
        "Does the blue plan cover cosmetic surgery?",
        "What will I pay if I have a hospital stay?",
        "is fertility treatment covered in any of the health plans?",
        "Does the blue plan cover cosmetic surgery?",
        "how much is a vision exam?",
        "how often can I have a vision exam?",
        "What are the bi-weekly medical rates?",
        "How much is preventive care?",
        "how much are dental fillings?",
        "Do HSA funds roll over?",
        "What is the prescription drug coverage?",
    ]
    for question in test_questions:
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


LangChainInstrumentor().instrument()

register_otel(
    endpoints=Endpoints.ARIZE,
    space_key=os.environ["ARIZE_SPACE_KEY"],
    api_key=os.environ["ARIZE_OTEL_API_KEY"],
    model_id="benefits-rag",
)


# rag_chain = rag_with_sources(
#     llm="gpt-4o",
#     index_name="benefits-rag",
#     llm_temperature=0,
# )


def few_shots():
    return [
        "If you have a hospital stay, the costs you will incur depend on whether your provider is in-network or out-of-network. Here are the details for each plan:\n\n### 2024 White Plan:\n- **In-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 20% coinsurance\n  - **Physician/surgeon fees:** 20% coinsurance\n- **Out-of-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 40% coinsurance\n  - **Physician/surgeon fees:** 40% coinsurance\n- **Limitations, Exceptions, & Other Important Information:** Prior Authorization required. Your cost share may increase to 50% if not obtained.\n\n**Source:** 2024 White Plan Summary of Benefits, Page 3, [URL](https://lpcorp.com/)\n\n### 2024 Orange Plan:\n- **In-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 20% coinsurance\n  - **Physician/surgeon fees:** 20% coinsurance\n- **Out-of-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 40% coinsurance\n  - **Physician/surgeon fees:** 40% coinsurance\n- **Limitations, Exceptions, & Other Important Information:** Prior Authorization required. Your cost share may increase to 50% if not obtained.\n\n**Source:** 2024 Orange Plan Summary of Benefits, Page 3, [URL](https://lpcorp.com/)\n\n### 2024 Blue Plan:\n- **In-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 20% coinsurance\n  - **Physician/surgeon fees:** 20% coinsurance\n- **Out-of-Network Provider:**\n  - **Facility fee (e.g., hospital room):** 40% coinsurance\n  - **Physician/surgeon fees:** 40% coinsurance\n- **Limitations, Exceptions, & Other Important Information:** Prior Authorization required. Your cost share may increase to 50% if not obtained.\n\n**Source:** 2024 Blue Plan Summary of Benefits, Page 3, [URL](https://lpcorp.com/)\n\nIn summary, for all plans, you will pay 20% coinsurance for in-network providers and 40% coinsurance for out-of-network providers. Prior authorization is required, and failure to obtain it may increase your cost share to 50%.",
        "The out-of-pocket maximum is the most you will have to pay for covered services in a year. Here are the details for each of LP's medical plans:\n\n### 2024 White Plan:\n- **In-Network:** $2,000 per person / $4,000 per family\n- **Out-of-Network:** $4,000 per person / $8,000 per family\n\n**Source:** 2024 White Plan Summary of Benefits, Page 1, [URL](https://lpcorp.com/)\n\n### 2024 Orange Plan:\n- **In-Network:** $4,000 per person\n- **Out-of-Network:** $8,000 per person\n\n**Source:** 2024 Orange Plan Summary of Benefits, Page 1, [URL](https://lpcorp.com/)\n\n### 2024 Blue Plan:\n- **In-Network:** $7,000 per person / $14,000 per family\n- **Out-of-Network:** $14,000 per person / $28,000 per family\n\n**Source:** 2024 Blue Plan Summary of Benefits, Page 1, [URL](https://lpcorp.com/)\n\nPlease note that premiums, balance-billing charges, penalties, and healthcare services not covered by the plan do not count towards the out-of-pocket maximum. Once you reach the out-of-pocket maximum, the insurance company will pay 100% of your eligible medical expenses for the rest of the year.",
        "For dental fillings under LP's dental plan, the coverage details are as follows:\n\n### In-Network (Delta Dental PPO):\n- **Basic Restorative (fillings, simple extraction, root canals):** Plan pays 80% after the deductible.\n\n### Out-of-Network:\n- **Basic Restorative (fillings, simple extraction, root canals):** Plan pays 80% after the deductible.\n\n### Additional Details:\n- **Annual Deductible:** $50 per individual / $150 per family.\n- **Calendar Year Maximum:** $1,500 per member.\n\nSo, for dental fillings, you will pay 20% of the cost after meeting your annual deductible, whether you go to an in-network or out-of-network provider.\n\n**Sources:**\n- 2024 LP Benefits Guide, Page 16, [URL](https://lpcorp.com/)\n- Dental Summary, Page 1-2, [URL](https://lpcorp.com/)",
    ]


questions = [
    "What is the deductible for the Orange Plan?",
    "what is the deductible for the white plan?",
    "What's the annual deductible in the blue plan?",
    "what is the out of pocket limit for the blue plan?",
    "what is the out of pocket limit for the white plan?",
    "what is the out of pocket limit for the orange plan?",
]
rag_chain = rag_with_alternatives(
    # llm="llama_405b_3_1",
    llm="gpt-4o",
    index_name="benefits-rag",
    llm_temperature=0,
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
