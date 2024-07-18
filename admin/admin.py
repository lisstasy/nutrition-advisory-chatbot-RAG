import boto3
import botocore 
import streamlit as st
import os
import uuid

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Initialize S3 and Bedrock clients
s3_client = boto3.client("s3")
bedrock_client = boto3.client(service_name='bedrock-runtime',
                              #region_name=os.environ.get("AWS_DEFAULT_REGION", None),
                              #config=botocore.config.Config(read_timeout=300,retries={'max_attemps':3})
                              )

# Initialize Bedrock embeddings model
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def get_unique_id():
    """Generate a unique request ID."""
    return str(uuid.uuid4())

def load_and_split_pdfs(uploaded_files):
    """Load PDFs, combine their pages, and split the combined text into chunks."""
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", "\t", " "]
    )
    documents = {}

    for uploaded_file in uploaded_files:
        source_filename = uploaded_file.name
        with open(source_filename, mode="wb") as w:
            w.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader(source_filename)
        pages = loader.load_and_split()

        combined_text = "\n\n".join([page.page_content for page in pages])
        documents[source_filename] = combined_text

    all_docs = []
    for source, combined_text in documents.items():
        chunks = text_splitter.split_text(combined_text)
        for chunk in chunks:
            metadata = {"source": source}
            all_docs.append(Document(page_content=chunk, metadata=metadata))

    return all_docs

def create_vector_store(request_id, documents):
    """Create and save a vector store from the given documents."""
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)
    file_name = f"{request_id}.bin"
    folder_path = "/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    bucket_name = os.environ.get("BUCKET_NAME")
    s3_client.upload_file(Filename=f"{folder_path}/{file_name}.faiss", Bucket=bucket_name, Key=f"{request_id}.faiss")
    s3_client.upload_file(Filename=f"{folder_path}/{file_name}.pkl", Bucket=bucket_name, Key=f"{request_id}.pkl")

    return True

def main():
    """Main function for the Streamlit app."""
    st.write("This is Admin Site for Chat with PDF demo")
    uploaded_files = st.file_uploader("Choose files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        documents = load_and_split_pdfs(uploaded_files)

        request_id = get_unique_id()
        st.write(f"Request Id: {request_id}")
        st.write(f"Total chunks: {len(documents)}")
        st.write("===================")

        if documents:
            st.write(f"The first chunk:\n{documents[0]}")
            st.write("-------------------")
            st.write(f"The last chunk:\n{documents[-1]}")

        st.write("===================")

        st.write("Creating the Vector Store")
        result = create_vector_store(request_id, documents)

        if result:
            st.write("Vectore Store created successfully!")
        else:
            st.write("Error!! Please check logs.")

if __name__ == "__main__":
    main()
