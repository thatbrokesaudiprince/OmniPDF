import streamlit as st
import os
from utils import *  # Import functions from utils.py
from api_utils import *  # Import functions from api_utils.py
import json
st.set_page_config(layout="wide", page_title="OmniPDF")

def main():

    # PDF upload and processing
    st.title("PDF Summarization App")
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        file_path = f"data/{uploaded_file.name}"

        # Save uploaded file
        with open(file_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())
            st.info("Uploaded PDF File")

        # Extract text and tables from the PDF
        pages, tables = extract_pdf_text_tables(file_path)
        translated_tables = []
        translated_text = []


        # Display PDF and extracted content in a two-column layout
        original_pdf_column, functionalities_column = st.columns(2)

        with original_pdf_column:
            display_pdf(file_path)  # Display uploaded PDF

        with functionalities_column:
            translate_tab, summary_tab, chat_tab = st.tabs(["Translation text to English", "View Summary","Chat with Omni"])
            with translate_tab:
                vernacular_text = st.text_area("Enter text to translate to english")
                if st.button("Translate Page/Table") and vernacular_text:
                        with st.spinner("Translating text into English"):
                            translated_text = translate_text_api(vernacular_text)['translation']
                        st.markdown(translated_text)

            with summary_tab:
                if st.button("Summarize"):
                    return  # Summarize functionality (to be implemented)
                
            with chat_tab:
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ask me anything your document!"):
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})

                response = f"Echo: {prompt}"
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
       
        with st.expander("View Tables"):
            extracted_tables_col, translated_tables_col = st.columns(2)
            with extracted_tables_col:
                st.info("Extracted Tables")
                for table in tables:
                    for t in table:
                        st.dataframe(t)

            with translated_tables_col:
                st.info("Translated Tables")
                for table in tables:
                    for t in table:
                        translated_table = translate_table_api(process_table_data(t))['translated_table']
                        st.dataframe(translated_table)
                        translated_tables.append(translated_table)

if __name__ == '__main__':
    main()
