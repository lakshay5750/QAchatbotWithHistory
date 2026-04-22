import streamlit as st
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ----------------------------
# CACHE EMBEDDINGS (IMPORTANT)
# ----------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

embeddings = get_embeddings()

# ----------------------------
# CACHE VECTOR STORE
# ----------------------------
@st.cache_resource
def create_vector_store(_splits):
    return Chroma.from_documents(
        documents=_splits,
        embedding=embeddings
    )

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("📄 Conversational RAG with PDF + Chat History")
st.write("Upload PDFs and chat with them like a smart assistant 🤖")

api_key = st.text_input("Enter Groq API Key", type="password")

if api_key:

    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile"
    )

    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    # ----------------------------
    # PROCESS PDF ONLY ONCE
    # ----------------------------
    if uploaded_files and "vector_store" not in st.session_state:

        documents = []

        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp_{uploaded_file.name}"

            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

            # DELETE TEMP FILE (IMPORTANT)
            os.remove(temp_pdf)

        # ----------------------------
        # CHUNKING (OPTIMIZED)
        # ----------------------------
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        splits = text_splitter.split_documents(documents)

        # ----------------------------
        # VECTOR STORE (CACHED)
        # ----------------------------
        st.session_state.vector_store = create_vector_store(splits)

    # ----------------------------
    # RUN ONLY IF VECTOR STORE READY
    # ----------------------------
    if "vector_store" in st.session_state:

        retriever = st.session_state.vector_store.as_retriever()

        # Contextualize Prompt
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given chat history and user question, rewrite it as a standalone question. "
             "Do NOT answer, only rewrite."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_prompt
        )

        # QA Prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful assistant. Use the context to answer. "
             "If answer not found, say 'I don't know'."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion:\n{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        # ----------------------------
        # SESSION MEMORY
        # ----------------------------
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # ----------------------------
        # CHAT UI
        # ----------------------------
        user_input = st.text_input("Ask your question:")

        if user_input:
            session_history = get_session_history(session_id)

            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            st.success(response["answer"])

            st.write("🧠 Chat History:")
            st.write(session_history.messages)

else:
    st.warning("Please enter Groq API key first ⚠️")