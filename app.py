import streamlit as st

def main():
    st.set_page_config(page_title="ChatPDFs", page_icon=":books:")
    
    st.header("ChatPDFs :books:")
    st.text_input("Ask a question about your document:")

    with st.sidebar:
        st.subheader("Your documents")

if __name__ == '__main__':
    main()