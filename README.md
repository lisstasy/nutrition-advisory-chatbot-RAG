# Nutrition Advisor Conversational Chatbot

Welcome to the Nutrition Advisor Chatbot project! This chatbot is designed to provide science-backed advice and tips based on insights from Dr. Will Bulsiewicz's book "Fiber Fueled". 

## Project Summary

The history-aware conversational chatbot is powered by Retrieval-Augmented Generation (RAG). The chatbot is designed to engage users in discussions about nutrition, gut health, and biotics (prebiotics, probiotics, postbiotics). It consists of two applications (ADMIN and USER) deployed using Docker containers for scalability and consistency across environments and Streamlit demos. It is built using a combination of AWS services (Amazon Bedrock and Amazon S3), FAISS with mmr reranking approach for vector storage and retrieval, LangChain’s conversational and memory chains, Docker for containerization, and Streamlit for interactive demos.

[![GIF](https://github.com/lisstasy/nutrition-advisory-chatbot-RAG/blob/main/index.html)]

P.S. You can adjust this project for your needs by uploading your PDF files via the Admin application, and modifying prompts in a user.py file.

## Key Features

### ADMIN Application

- **PDF Upload:** Web application allowing admin users to upload PDF documents.
- **Text Processing:** Splitting PDF text into manageable chunks for processing.
- **Vector Embeddings:** Leveraged the Amazon Titan Embedding Model via Amazon Bedrock to create vector representations of text chunks.
- **Indexing:** Used FAISS for vector indexing and stored the index locally before uploading to Amazon S3 for persistence.

### USER Application

- **Query Interface:** Web interface enabling users to interact and query the chatbot.
- **Index Management:** Downloaded the FAISS index from Amazon S3 to establish a local vector store upon application start.
- **Conversational AI:** Utilized Langchain's retrieval and memory chains to:
  - Convert user queries into vector embeddings using the Amazon Titan Embedding Model.
  - Perform a maximum marginal relevance search in the FAISS index to retrieve relevant documents.
  - Use a prompt template to provide the LLM (Mixtral-8x7B Instruct model from Mistral via Amazon Bedrock) with the user's query rephrased based on the chat history and relevant context.
- **User Interaction:** Displayed the LLM's responses within the chat interface.

## Tech Stack

- **Amazon Bedrock:** Used for the Amazon Titan Embedding Model and access to the Mixtral-8x7B Instruct model from Mistral.
- **Amazon S3:** Stored and retrieved the FAISS index files for persistent data management.
- **FAISS:** Applied for efficient vector indexing and mmr searches.
- **Langchain:** Utilized for managing conversation retrieval and summary memory.
- **Docker:** Deployed both ADMIN and USER applications within Docker containers for seamless deployment and scalability.
- **Streamlit:** Interactive Demos for admin and users to interact with the chatbot, making it easy to demonstrate and test the application's capabilities.


## Project Structure
```
.
├── admin
│   ├── .env
│   ├── admin.py
│   ├── Dockerfile
│   └── requirements.txt
├── docs
├── rag-env
├── user
    ├── .env
    ├── Dockerfile
    ├── requirements.txt
    └── user.py
```
    
## How to Run the Project

### 1. Clone the Repository

```
git clone https://github.com/lisstasy/nutrition-advisory-chatbot-RAG.git
cd nutrition-chatbot
```
### 2. Set Up Environment Variables
Create and configure your .env files in both the admin and user directories with the necessary AWS credentials and other configuration settings.

```
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=your_aws_region
BUCKET_NAME=your_bucket_name
```

### 3. Build and Run Docker Containers
- For Admin Interface
Navigate to the admin directory, next, build the Docker Image, lastly, run the Docker Container

```
cd admin
docker build -t nutri-rag-admin .
docker run --env-file .env -v ~/.aws:/root/.aws -p 8083:8083 -it nutri-rag-admin
```

- For User Interface
Navigate to the user directory, next, build the Docker Image and run the Docker Container

```
cd user
docker build -t nutri-rag-user .
docker run --env-file .env -v ~/.aws:/root/.aws -p 8084:8084 -it nutri-rag-user
```

### 4. Access the Application
- Admin Interface: Open your browser and navigate to http://localhost:8083.
- User Interface: Open your browser and navigate to http://localhost:8084.



