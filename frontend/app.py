import streamlit as st
import os
from utils import *  # Import functions from utils.py
import json
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from classes.WordCloudGenerator import WordCloudGenerator
from classes.PDFProcessor import PDFProcessor
from classes.TableDataProcessor import TableDataProcessor
from classes.RAGHelper import RAGHelper
from classes.APIRouter import (
    translate_table,
    translate_text,
    summarize_table,
    summarize_text,
    caption_image,
    CLIENT,
)

st.set_page_config(layout="wide", page_title="OmniPDF")

wcg = WordCloudGenerator()
tbp = TableDataProcessor()
rh = RAGHelper()


def main():
    st.title("OmniPDF: Your PDF Assistant 🦸")
    st.subheader("Upload your PDF file and explore its content")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        file_path = f"data/{uploaded_file.name}"

        # Save file only if not already processed
        if "pdf_file" not in st.session_state or st.session_state.pdf_file != file_path:
            with open(file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
                st.session_state.pdf_file = file_path

            pdf = PDFProcessor(file_path)
            st.session_state.PAGES_DATA = pdf.get_all_data()
            st.session_state.TRANSLATED_TEXT = ""
            st.session_state.TRANSLATED_TABLES = []
            st.session_state.ALL_TEXT = "".join(
                page.get("text", "") for page in st.session_state.PAGES_DATA
            )
            st.session_state.IMAGE_CAPTIONS = {}

        ALL_TEXT = st.session_state.ALL_TEXT
        PAGES_DATA = st.session_state.PAGES_DATA
        TRANSLATED_TEXT = st.session_state.TRANSLATED_TEXT
        TRANSLATED_TABLES = st.session_state.TRANSLATED_TABLES
        IMAGE_CAPTIONS = st.session_state.IMAGE_CAPTIONS

        original_pdf_column, functionalities_column = st.columns(2)

        with original_pdf_column:
            pdf_display = display_pdf(st.session_state.pdf_file)
            st.session_state.pdf_display = pdf_display
            st.markdown(pdf_display, unsafe_allow_html=True)

        with functionalities_column:
            translate_tab, summary_tab, chat_tab = st.tabs(
                ["Translate Text", "View Summary", "Chat with Omni"]
            )

            with translate_tab:
                vernacular_text = st.text_area(
                    "Enter text to translate to English",
                    st.session_state.TRANSLATED_TEXT,
                )
                if st.button("Translate Page/Table") and vernacular_text:
                    with st.spinner("Translating text..."):
                        st.session_state.TRANSLATED_TEXT = translate_text(
                            vernacular_text, CLIENT
                        )
                st.markdown(st.session_state.TRANSLATED_TEXT)

            with summary_tab:
                if st.button("Summarize"):
                    st.info("Summary functionality is not yet implemented.")

            with chat_tab:
                if st.button("Embed Text") and ALL_TEXT:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)
                    docs = text_splitter.split_text(ALL_TEXT)

                    # Insert the Documents as embeddings to the vector database
                    # Translate text to English
                    doc_data = []
                    for i, doc in enumerate(docs, start=1):
                        translated_text = translate_text(doc, CLIENT)
                        doc_data.append(
                            Document(
                                page_content=translated_text,
                                metadata={
                                    "chunk_index": str(i),
                                    "source": uploaded_file.name,
                                    "type": "text",
                                },
                            )
                        )
                    rh.add_docs_to_chromadb(doc_data)
                    docs = rh.retrieve_relevant_docs("Summarize this")
                    print(docs)

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ask me anything about your document!"):
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt}
                    )
                    response = f"Echo: {prompt}"
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    with st.chat_message("assistant"):
                        st.markdown(response)

        with st.expander("View Tables"):
            extracted_tables_col, TRANSLATED_TABLES_col = st.columns(2)

            with extracted_tables_col:
                for page in PAGES_DATA:
                    for table in page.get("tables", []):
                        st.write(f"Table on Page {page.get('page_number')}")
                        st.dataframe(table)

            with TRANSLATED_TABLES_col:
                if st.button("Translate Tables"):
                    st.session_state.TRANSLATED_TABLES = []
                    for page in PAGES_DATA:
                        for table in page.get("tables", []):
                            translated_table = translate_table(
                                tbp.format_for_json(table), CLIENT
                            )

                            st.session_state.TRANSLATED_TABLES.append(translated_table)
                            st.write(f"Table on Page {page.get('page_number')}")
                            st.dataframe(translated_table)

        with st.expander("View WordCloud"):
            maxwords = st.number_input(
                "Top words to display", min_value=1, max_value=1000, value=100
            )
            st.info(f"Displaying top {maxwords} words")
            wordcloud = wcg.generate_wordcloud(
                text=ALL_TEXT,
                max_words=maxwords,
                height=200,
                width=400,
            )
            st.pyplot(wordcloud, use_container_width=True)

        with st.expander("View Images"):
            for page in PAGES_DATA:
                for img in page.get("images", []):
                    try:
                        caption = caption_image(img, CLIENT)
                    except:
                        caption = "Failed to caption"
                    IMAGE_CAPTIONS[img] = caption
                    st.image(
                        img,
                        caption=f"{caption}\nFound on Page: {page.get('page_number')}",
                    )


if __name__ == "__main__":
    main()
