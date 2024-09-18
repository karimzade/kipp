from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
import os
import streamlit as st
import tempfile
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.chains.question_answering import load_qa_chain
import os
from dotenv import load_dotenv
load_dotenv()


def partition_file(input_path):
    loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".docx": Docx2txtLoader,
        ".md": UnstructuredMarkdownLoader,
    }
    file_extension = os.path.splitext(input_path)[1].lower()
    loader_class = loaders.get(file_extension)
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_extension}")
    loader = loader_class(input_path)
    documents = loader.load_and_split() if file_extension == ".pdf" else loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if not client.list_collections():
        client.create_collection("consent_collection")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory="./chroma_db",
    )
    vectordb.persist()
    return vectordb


def create_llm(model_name):
    llm = OllamaLLM(model=model_name)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain


def get_llm_response(query, vectordb, chain):
    matching_docs = vectordb.similarity_search(
        query,
        k=10,
    )
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


st.title("Kipp: Chat with Documents")
uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True)
model = "llama3.1:8b"

if uploaded_files:
    folder_path = tempfile.mkdtemp()
    os.makedirs(folder_path, exist_ok=True)

    for uploaded_file in uploaded_files:
        path = os.path.join(folder_path, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    vectordb = partition_file(path)
    if vectordb:
        chain = create_llm(model)
        st.session_state.vectordb = vectordb
        st.session_state.chain = chain
    else:
        st.error(
            "Failed to create vector database. Please check the logs for more details."
        )

    if "vectordb" in st.session_state and "chain" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask a question:")

        if prompt:
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            response = get_llm_response(
                prompt, st.session_state.vectordb, st.session_state.chain
            )
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
else:
    st.warning("Please upload and process your documents first.")
