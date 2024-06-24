import itertools as it
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from unstructured.staging.base import dict_to_elements
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError

load_dotenv()


gpt_4o = init_chat_model(model="gpt-4o", model_provider="openai", temperature=0)


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
    # EcxV76IXM7JZ6r5KtKqI63hOLXHGph
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
    req = shared.PartitionParameters(
        files=files,
        strategy="hi_res",
        hi_res_model_name="yolox",
        pdf_infer_table_structure=True,
        skip_infer_table_types=[],
    )
    try:
        resp = client.general.partition(request=req)
        pdf_elements = dict_to_elements(resp.elements)
    except SDKError as e:
        print(e)
    else:
        print(f"FINISHED PROCESSING {file_name}")
        return pdf_elements


def get_metadata(element):
    metadata = element.metadata.to_dict()
    metadata["languages"] = ", ".join(metadata.pop("languages", ""))
    metadata["category"] = element.category
    metadata["url"] = "https://lpcorp.com/"
    metadata["file_name"] = Path(metadata.pop("filename")).stem
    return metadata


def get_document(element):
    metadata = get_metadata(element=element)
    return Document(page_content=element.text, metadata=metadata)


def create_langchain_document(elements):
    return [
        get_document(element)
        for element in elements
        if element.category not in ("Title")
    ]


def get_splits(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    langchain_document = create_langchain_document(document)
    return text_splitter.split_documents(langchain_document)


def split_docs(documents):
    return list(it.chain.from_iterable(get_splits(document) for document in documents))


def add_records_to_pinecone_vs(documents, embedding, index_name):
    return PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embedding,
        index_name=index_name,
    )


open_ai_embedding = OpenAIEmbeddings(model="text-embedding-3-small")


pinecone_index = "lp-hr-rag"
# create_pinecone_index(
#     index_name=pinecone_index,
#     dimension=1536,
# )


files = Path("").glob("*.pdf")
# pdfs = [get_unstructured_pdf_elements(pdf) for pdf in pdfs]
# docs = split_docs(pdfs)

# %%
# add_records_to_pinecone_vs(
#     documents=docs, embedding=open_ai_embedding, index_name="lp-hr-rag"
# )


# docs = split_docs(unstructured_docs)
