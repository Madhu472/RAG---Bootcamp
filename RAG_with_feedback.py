# Importing Necessary Packages 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import Config
import pandas as pd
from datetime import datetime
import streamlit as st
import os
import warnings
warnings.filterwarnings("ignore")

# Loading Secret Keys - Environmental Variables
os.environ["OPENAI_API_TYPE"] = Config.key_value_dict['api_type']
os.environ["OPENAI_API_VERSION"] = Config.key_value_dict['api_version']
os.environ["OPENAI_API_BASE"] = Config.key_value_dict['api_base']
os.environ["OPENAI_API_KEY"] = Config.key_value_dict['api_key']

# Function for Feedback capture
def on_submit(question, answer, feedback):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame(
        [[timestamp, question, answer, feedback]],
        columns=["timestamp", "question", "answer", "feedback"],
    )
    if not os.path.isfile("feedback.csv"):
        df.to_csv("feedback.csv", header="column_names",index = False)
    else:
        df.to_csv("feedback.csv", mode="a", header=False,index = False)


# Seesion Variables for Streamlit - Feedback capture session Variables
if "message" not in st.session_state:
    st.session_state.message = []
if "feedback" not in st.session_state:
    st.session_state.feedback = None

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = None

if "thumbs_up" not in st.session_state:
    st.session_state.thumbs_up = None

if "thumbs_down" not in st.session_state:
    st.session_state.thumbs_down = None

if "question_asked" not in st.session_state:
    st.session_state.question_asked = None

# Function to read the Input PDF Documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Chunking the input text
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return chunks

# Embedding function 
def get_vector_store(chunks):
    embeddings = AzureOpenAIEmbeddings(
        deployment=Config.key_value_dict["embed_eng_dep_nm"],
        model=Config.key_value_dict["embedding_model"],
        chunk_size=1,
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Retrival Chain for responses with prompt
def qachain():
    llm = AzureChatOpenAI(
        deployment_name=Config.key_value_dict["comp_eng_dep_nm"],
        temperature=0,
        openai_api_version=Config.key_value_dict["api_version"], max_tokens=4000)

    prompt_template = """ You are an AI Chatbot assistant trained on the context provided. 
    Use the following pieces of context to answer the question:
    Provide me all the steps involved in resolving/guiding with clear explanation, 
    Paraphrase the steps to fluent english so that it feels like chatting with a human
    Always answer in way that you are creator and holds responsible always in providing steps with proper formatting,
    Give me answer in clear number wise steps always,
    Strictly you don't know the answer, just say that Out of Context, Ask questions from the content uploaded for futher help in assistance.
    
    {context}
    
    Question: {question}

    Answer: """
    
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    # Running the LLM - Chain
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# User Interaction with the app
def user_input(user_question):
    embeddings = AzureOpenAIEmbeddings(
        deployment=Config.key_value_dict["embed_eng_dep_nm"],
        model=Config.key_value_dict["embedding_model"],
        chunk_size=1,
    )
    db = FAISS.load_local("faiss_index", embeddings)
    docs = db.similarity_search(user_question)
    chain = qachain()
    result = chain.run(input_documents=docs, question=user_question)
    return result

# Clear chat history
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

# Function Invoking Web app
def main():
    st.set_page_config(
        page_title="PDF Chatbot",
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Gen AI Powered App")
        pdf_docs = st.file_uploader(
            "Upload PDF Files and Click on the Submit Button", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files..!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload the pdf and ask me a question relating to it...!"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.question_asked = prompt

        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
    
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                placeholder.markdown(response)
    
        if response is not None:
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)

        # new code
        st.session_state.answer = response
        # Reset variables
        st.session_state.feedback = None
        st.session_state.feedback_submitted = None
        st.session_state.thumbs_up = None
        st.session_state.thumbs_down = None

    if st.session_state.question_asked is not None:
        st.session_state.feedback_submitted = None
        st.session_state.thumbs_up = None
        st.session_state.thumbs_down = None
        # Feedback whether new question or old for answer
        col1, col2, col3 = st.columns([9, .8, 1])
        thumbs_up = col2.button("👍")
    
        if thumbs_up:
            on_submit(
                st.session_state.question_asked, st.session_state.answer, "Positive"
            )
            st.session_state.question_asked = None
            st.success("Thank you for the feedback..!")
            st.rerun()
        thumbs_down = col3.button("👎")

        if thumbs_down:
            on_submit(
                st.session_state.question_asked, st.session_state.answer, "Negative"
            )
            st.session_state.question_asked = None
            st.success("Thank you for the feedback..!")
            st.rerun()

    elif st.session_state.feedback_submitted is not None:
        # Display "Thank you for your feedback!" message
        st.success("Thank you for your feedback!")

    else:
        # Fixes random error of feedback button showing
        st.session_state.feedback_submitted = None

if __name__ == "__main__":
    main()