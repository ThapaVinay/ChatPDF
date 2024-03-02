import streamlit as st
from dotenv import load_dotenv


def main():
    load_dotenv()
    st.set_page_config(page_title="ChatPDFs", page_icon=":books:")
    
    st.header("ChatPDFs :books:")
    st.text_input("Ask a question about your document:")

    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload your PDFs here and click on 'Process'")
        st.button("Process")

if __name__ == '__main__':
    main()