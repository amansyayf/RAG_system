from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd


def data_ingestion():
    knowledge_data = pd.read_csv('../data/knowledge_base_cleaned_.csv')
    chunks = knowledge_data.chunk.tolist()
    documents = []

    for item in range(len(chunks)):
        page = Document(page_content=chunks[item], )
        documents.append(page)

    return documents


def get_vector_store(docs):
    model_name = "intfloat/multilingual-e5-large"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store_faiss = FAISS.from_documents(docs, embedding=embeddings)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss


if __name__ == '__main__':
    docs = data_ingestion()
    get_vector_store(docs)
