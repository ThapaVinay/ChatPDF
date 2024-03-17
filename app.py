import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from htmlTemplate import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import re

# get the pdf document
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# break the text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    return chunks

# get the vector embeddings
def get_vectorstore(text_chunks):
    inference_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key = inference_api_key, model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

# make the converstion chain
def get_conversation_chain(vectorstore):
    
    llm = HuggingFaceHub( repo_id="mistralai/Mistral-7B-Instruct-v0.2", model_kwargs={"temperature": 0.5, "max_length": 64, "max_new_tokens":512})
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
    )
    return conversation_chain

# handle the user input
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question, 'chat_history': []})
    
    match = re.findall(r'\nHelpful Answer: (.*)', response['answer'])

    st.write(user_template.replace(
                "{{MSG}}", user_question), unsafe_allow_html=True)
    st.write(bot_template.replace(
                "{{MSG}}", str([item for item in match])), unsafe_allow_html=True)

# main function
def main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot", page_icon=":rockets:")
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("ChatBot 😁")
    
    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()