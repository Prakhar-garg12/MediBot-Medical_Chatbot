from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore 
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os


load_dotenv()

pc = Pinecone(
        api_key="PINECONE_API_KEY"
    )

index_name = 'medical'

extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

vectorstore_from_texts = PineconeVectorStore.from_texts(
        [t.page_content for t in text_chunks],
        index_name=index_name,
        embedding=embeddings
    )