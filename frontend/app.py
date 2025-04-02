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
    rag_prompt,
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
            # st.session_state.TRANSLATED_TEXT = ""
            # st.session_state.TRANSLATED_TABLES = []
            st.session_state.ALL_TEXT = "".join(
                page.get("text", "") for page in st.session_state.PAGES_DATA
            )
            st.session_state.ALL_TEXT_TRANSLATED = "".join(
                page.get("translated_text", "") for page in st.session_state.PAGES_DATA
            )

        original_pdf_column, functionalities_column = st.columns(2)

        with original_pdf_column:
            pdf_display = display_pdf(st.session_state.pdf_file)
            st.session_state.pdf_display = pdf_display
            st.markdown(pdf_display, unsafe_allow_html=True)

        with functionalities_column:
            translate_tab, chat_tab = st.tabs(["Translate Text", "Chat with Omni"])

            with translate_tab:
                vernacular_text = st.text_area(
                    "Enter text to translate to English",
                    # st.session_state.TRANSLATED_TEXT,
                )
                if st.button("Translate Page/Table") and vernacular_text:
                    with st.spinner("Translating text..."):
                        translation_response = translate_text(vernacular_text, CLIENT)
                    st.markdown(translation_response)

            with chat_tab:
                if st.button("Embed Text") and st.session_state.ALL_TEXT:
                    # Split text into chunks of 1024
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)
                    text_docs = text_splitter.split_text(
                        st.session_state.ALL_TEXT_TRANSLATED
                    )

                    # Insert the Documents as embeddings to the vector database
                    doc_data = []

                    # Text Documents
                    for i, doc in enumerate(text_docs, start=1):
                        # Translate text to English
                        doc_data.append(
                            Document(
                                page_content=doc,
                                metadata={
                                    "chunk_index": str(i),
                                    "source": uploaded_file.name,
                                    "type": "text",
                                },
                            )
                        )

                    # Table Documents

                    for page in st.session_state.PAGES_DATA:
                        for table in page["translated_tables"]:
                            summary = table["translated_table_summary"]

                            doc_data.append(
                                Document(
                                    page_content=summary,
                                    metadata={
                                        "chunk_index": i,
                                        "source": uploaded_file.name,
                                        "type": "table",
                                        "table_content_key": i,
                                    },
                                )
                            )

                    # Image Documents
                    for i, page in enumerate(st.session_state.PAGES_DATA, start=1):
                        for image in page["images"]:
                            url = image["image_url"]
                            caption = image["caption"]

                        doc_data.append(
                            Document(
                                page_content=st.session_state.IMAGE_CAPTIONS[doc],
                                metadata={
                                    "chunk_index": str(i),
                                    "source": uploaded_file.name,
                                    "type": "image",
                                    "img_path": str(doc),
                                },
                            )
                        )
                    rh.add_docs_to_chromadb(doc_data)

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Create a scrollable chat container using st.container()
                chat_container = st.container()

                with chat_container:
                    message_placeholder = (
                        st.empty()
                    )  # Placeholder to store messages dynamically

                    # Display chat history inside a container
                    with message_placeholder.container():
                        for message in st.session_state.messages:
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])

                # Handle user input
                if prompt := st.chat_input("Ask me anything about your document!"):
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt}
                    )

                    rel_docs = rh.retrieve_relevant_docs(prompt)
                    response = rag_prompt(prompt, rel_docs, CLIENT)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )

                    # Update chat display dynamically
                    with message_placeholder.container():
                        for message in st.session_state.messages:
                            with st.chat_message(message["role"]):
                                st.markdown(message["content"])

                        for doc in rel_docs:
                            if doc.metadata.get("type") == "text":
                                st.markdown(doc.page_content)
                            elif doc.metadata.get("type") == "table":
                                st.write("Table Data:")
                                print(doc)
                                print(st.session_state.translated_tables2)
                                print(doc.page_content)
                                st.dataframe(
                                    st.session_state.translated_tables2[
                                        doc.metadata.get("table_content_key")
                                    ]
                                )
                            elif doc.metadata.get("type") == "image":
                                st.image(
                                    doc.metadata["img_path"],
                                    caption=doc.page_content,
                                    use_column_width=True,
                                )

        with st.expander("View Tables"):
            extracted_tables_col, TRANSLATED_TABLES_col = st.columns(2)

            with extracted_tables_col:
                for page in st.session_state.PAGES_DATA:
                    for table in page.get("tables", []):
                        st.write(f"Table on Page {page.get('page_number')}")
                        st.dataframe(table)

            with TRANSLATED_TABLES_col:
                for page in st.session_state.PAGES_DATA:
                    for table in page.get("translated_tables", []):
                        st.write(f"Table on Page {page.get('page_number')}")
                        st.dataframe(table)

        with st.expander("View WordCloud"):
            maxwords = st.number_input(
                "Top words to display", min_value=1, max_value=1000, value=100
            )
            st.info(f"Displaying top {maxwords} words")
            wordcloud = wcg.generate_wordcloud(
                text=st.session_state.ALL_TEXT,
                max_words=maxwords,
                height=200,
                width=400,
            )
            translated_wordcloud = wcg.generate_wordcloud(
                text=st.session_state.ALL_TEXT_TRANSLATED,
                max_words=maxwords,
                height=200,
                width=400,
            )
            st.pyplot(wordcloud, use_container_width=True)
            st.pyplot(translated_wordcloud)

        with st.expander("View Images"):
            for page in st.session_state.PAGES_DATA:
                for image in page.get("images", []):
                    caption = image["caption"]
                    st.image(
                        image["image_url"],
                        caption=f"{caption}\nFound on Page: {page.get('page_number')}",
                    )

        print(st.session_state.PAGES_DATA)


if __name__ == "__main__":
    main()
