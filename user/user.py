import boto3
import streamlit as st
import os
import uuid
import time
from dotenv import load_dotenv

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Initialize AWS clients
s3_client = boto3.client("s3")
bedrock_client = boto3.client(service_name="bedrock-runtime")

# Configuration
bucket_name = os.environ.get("BUCKET_NAME")
folder_path = "/tmp/"

# Initialize session storage
store = {}

def ensure_folder_exists(path):
    os.makedirs(path, exist_ok=True)

def load_index(bucket_name, folder_path):
    ensure_folder_exists(folder_path)
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' not in response:
            st.warning("No files found in the bucket.")
            return
        
        for obj in response['Contents']:
            key = obj['Key']
            file_path = os.path.join(folder_path, key)
            s3_client.download_file(Bucket=bucket_name, Key=key, Filename=file_path)
            #st.info(f"Downloaded {key} to {file_path}")
    
    except Exception as e:
        st.error(f"Error loading index: {e}")

def get_llm():
    llm = Bedrock(
        model_id="mistral.mixtral-8x7b-instruct-v0:1", 
        client=bedrock_client,
        model_kwargs={"max_tokens": 512, "temperature": 0.1}
    )
    return llm

def get_response(llm, vectorstore, question, session_id="abc124"):
    # Configure retrieval and question-answering chains
    retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={"k": 4})

    rephrase_template = (
        "<s>[INST] Given a chat history and a follow up question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is.\n"
        "Chat History: {chat_history}\nFollow Up Input: {input}\nStandalone Question: [/INST]</s>"
    )
    rephrase_prompt = PromptTemplate(input_variables=["chat_history", "input"], template=rephrase_template)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, rephrase_prompt)

    system_prompt = (
        "<s>[INST] You are an gut health and plant-based nutrition "
        "assistant for question-answering tasks powered by the "
        "insights from Dr. Will Bulsiewicz's book 'Fiber Fueled'. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "Chat History: {chat_history}\nOther context: {context}\nQuestion: {input} [/INST]</s>"
    )
    qa_prompt = PromptTemplate(input_variables=["chat_history", "context", "input"], template=system_prompt)
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Statefully manage chat history
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    start_time = time.time()
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )
    end_time = time.time()
    elapsed_time = int((end_time - start_time) * 1000)
    # logging.info(f"LLM ({elapsed_time}ms): {response['answer']}")
    return response

def main():
    st.set_page_config(page_title="Nutrition Chatbot Demo")

    st.title("Nutrition Advisor Chatbot")
    st.write("Meet your personal guide to gut health and plant-based nutrition, powered by the insights from Dr. Will Bulsiewicz's book 'Fiber Fueled'. This bot provides science-backed advice and tips to optimize your gut microbiome. Whether you're looking to improve digestion, boost immunity, or simply eat healthier, your Nutrition Advisor is here to support your journey to a thriving gut and vibrant health.")

    # Load index files
    load_index(bucket_name, folder_path)

    # Initialize FAISS index and LLM model
    faiss_index = FAISS.load_local(
        index_name="5115693b-e483-48c6-8b45-a746ed4eeb20",
        folder_path=folder_path,
        embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client),
        allow_dangerous_deserialization=True
    )
    llm = get_llm()

    # Initialize session state
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello there, how can I help you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_input = st.chat_input("Ask a question")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(llm, faiss_index, user_input, session_id=st.session_state.session_id)
                ai_response = response['answer']
                st.write(ai_response)

        st.session_state.messages.append({"role": "assistant", "content": ai_response})

if __name__ == "__main__":
    main()
