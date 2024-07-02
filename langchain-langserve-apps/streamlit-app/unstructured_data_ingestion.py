import itertools as it
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_core.documents import Document

# from langchain.chat_models import init_chat_model
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve.schema import CustomUserType
from pinecone import Pinecone, ServerlessSpec
from langchain_cohere import CohereEmbeddings
from unstructured.staging.base import convert_to_dict
from pydantic import Field
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import dict_to_elements
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

load_dotenv()


def create_pinecone_index(index_name, dimension):
    pc = get_pinecone_client()
    spec = ServerlessSpec(
        cloud="aws",
        region="us-west-2",
    )

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, dimension=dimension, metric="cosine", spec=spec
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)


def get_pinecone_client():
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def get_unstructured_client():
    return UnstructuredClient(
        api_key_auth=os.environ["UNSTRUCTURED_API_KEY"],
        server_url="https://api.unstructuredapp.biz/general/v0/general",
    )


def get_unstructured_pdf_elements(file_path):
    client = get_unstructured_client()
    file_name = file_path.stem
    print(f"Processing {file_name}")
    with open(file_path := file_path.as_posix(), "rb") as f:
        files = shared.Files(
            content=f.read(),
            file_name=file_path,
        )
    request = shared.PartitionParameters(
        files=files,
        # strategy="fast",  # "hi_res",
        strategy="hi_res",  # "hi_res",
        hi_res_model_name="yolox",
        pdf_infer_table_structure=True,
        skip_infer_table_types=[],
        # extract_image_block_types=[],
        # chunking_strategy="by_title",  # "'basic', 'by_page', 'by_similarity' or 'by_title'"
    )
    try:
        response = client.general.partition(request=request)
        pdf_elements = dict_to_elements(response.elements)
    except SDKError as e:
        print(e)
    else:
        print(f"FINISHED PROCESSING {file_name}")
        return pdf_elements


def get_metadata(element, parent_ids):
    metadata = element.metadata.to_dict()
    metadata["languages"] = ", ".join(metadata.pop("languages", ""))
    metadata["category"] = element.category
    metadata["url"] = "https://lpcorp.com/"
    metadata["file_name"] = Path(metadata.pop("filename")).stem
    if metadata["category"] != "Title":
        metadata["parent_title"] = parent_ids.get(metadata.get("parent_id", ""), "")

    return metadata


def get_document(element, parent_ids_for_titles):
    metadata = get_metadata(element=element, parent_ids=parent_ids_for_titles)
    return Document(page_content=element.text, metadata=metadata)


def create_langchain_document(elements):
    parent_ids_for_titles = {
        element.id: element.text for element in elements if element.category == "Title"
    }
    return [
        get_document(element, parent_ids_for_titles)
        for element in elements
        if element.category not in ("Image")
    ]


def get_splits(document, semantic_chunking):
    if semantic_chunking:
        text_splitter = SemanticChunker(
            OpenAIEmbeddings(model="text-embedding-3-small")
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
    langchain_document = create_langchain_document(document)
    return text_splitter.split_documents(langchain_document)


def split_docs(documents, semantic_chunking):
    return list(
        it.chain.from_iterable(
            get_splits(document, semantic_chunking) for document in documents
        )
    )


def add_records_to_pinecone_vs(documents, embedding, index_name):
    return PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embedding,
        index_name=index_name,
    )


def crawl_webpage(url):
    app = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])
    return app.crawl_url(
        url=url,
        params={
            "crawlerOptions": {"limit": 10},
            "pageOptions": {"onlyMainContent": True},
        },
        wait_until_done=False,
    )


def scrape_url(url, llm_extraction=False):
    app = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])
    llm_params = {
        "extractorOptions": {
            "mode": "llm-extraction",
            "extractionPrompt": "Based on the information on the page, extract the information from the schema. ",
            "extractionSchema": {},
        }
    }
    return app.scrape_url(
        url=url,
        params=llm_params if llm_extraction else {},
    )


def exclude_images(documents):
    return [document for document in documents if document.category != "Image"]


def convert_unstructured_to_csv(file_names, unstructured_docs):
    unstructured_dicts = [
        convert_to_dict(unstructured_doc) for unstructured_doc in unstructured_docs
    ]
    for file, dict_ in zip(file_names, unstructured_dicts):
        file, *_ = file.name.split(".")
        with open(
            f"/Users/airjoshua/gen_ai/gen-ai/langchain/langchain-app/data/unstructured_docs{file}.json",
            "w",
        ) as write_file:
            json.dump(dict_, write_file)


def get_json_unstructured():
    with open("data_file.json", "r") as read_file:
        data = json.load(read_file)


def upload_to_pinecone():
    docs = split_docs(
        documents=None,
        semantic_chunking=False,
    )

    add_records_to_pinecone_vs(
        documents=docs,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        index_name="lp",
    )


def normalize_pdf():
    files = list(
        Path(
            "/Users/airjoshua/gen_ai/gen-ai/langchain/langchain-app/data/manufacturing"
        ).glob("*.pdf")
    )
    files = list(
        Path("/Users/airjoshua/gen_ai/gen-ai/langchain/langchain-app/data").glob(
            "*.pdf"
        )
    )

    efficiency = get_unstructured_pdf_elements(
        Path(
            "/langchain/langchain-app/data/manufacturing/BreakdownOfEfficiencySectionOnDailyProductionReportv2.pdf"
        )
    )
    pdfs = [get_unstructured_pdf_elements(pdf) for pdf in files]
    chunked_pdfs = [
        chunk_by_title(pdf, overlap=0, max_characters=1000, multipage_sections=False)
        for pdf in pdfs
    ]
    element_chunking = [chunk_elements(pdf, max_characters=1000) for pdf in pdfs]
    docs = split_docs(
        documents=chunked_pdfs,
        semantic_chunking=False,
    )
    add_records_to_pinecone_vs(
        documents=docs,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        index_name="lp",
    )
    no_images = [exclude_images(documents) for documents in pdfs]
    title_chunks = chunk_by_title()
    # how to post a job requistion

    no_images_chunked_pdfs = [
        chunk_by_title(pdf, overlap=50, max_characters=1000) for pdf in pdfs
    ]


# normalize_pdf()


# pinecone_index = "lp"
# create_pinecone_index(
#     index_name=pinecone_index,
#     dimension=1536,
# )
