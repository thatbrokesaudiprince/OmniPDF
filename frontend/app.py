import streamlit as st
import os
from utils import *  # Import functions from utils.py
from api_utils import *  # Import functions from api_utils.py
import json
st.set_page_config(layout="wide", page_title="DOC-TER")

def main():

    # PDF upload and processing
    st.title("PDF Summarization App")
    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        file_path = f"data/{uploaded_file.name}"
        # Extract text and tables from the PDF
        pages, tables = extract_pdf_text_tables(file_path)
        translated_tables = []
        translated_text = []

        # Save uploaded file
        with open(file_path, 'wb') as temp_file:
            temp_file.write(uploaded_file.read())
            st.info("Uploaded PDF File")

        # Display PDF and extracted content in a two-column layout
        col1, col2 = st.columns(2)

        with col1:
            original_pdf, extracted_images = st.tabs(["Original PDF", "Extracted Images"])
            with original_pdf:
                display_pdf(file_path)  # Display uploaded PDF

        with col2:
            translate_tab, summary_tab = st.tabs(["Translation text to English", "View Summary"])
            st.info("View Summarize")
            with translate_tab:
                if st.button("Translate Page/Table"):
                    return  # Translate functionality (to be implemented)

            with summary_tab:
                if st.button("Summarize"):
                    return  # Summarize functionality (to be implemented)
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
