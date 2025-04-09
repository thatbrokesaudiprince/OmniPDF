import base64
import io
import os
from itertools import cycle  # For displaying images in columns

import streamlit as st
import requests
from PIL import Image

from classes.WordCloudGenerator import WordCloudGenerator
from classes.TableDataProcessor import TableDataProcessor
from classes.DataPreparer import DataPreparer
from utils import *  # Import functions from utils.py

st.set_page_config(layout="wide", page_title="OmniPDF")

wcg = WordCloudGenerator()
tbp = TableDataProcessor()
dp = DataPreparer()


@st.fragment
def download_pdf_data(
    uploaded_file_name: str,
    page_data: list[dict],
    all_text: str,
    all_text_translated: str,
):
    """Prepare and download PDF in JSON format."""

    if st.button("Prepare JSON Data"):
        zipped_folderpath = dp.prepare_pdf_data(
            uploaded_file_name,
            page_data,
            all_text,
            all_text_translated,
        )
        with open(f"{zipped_folderpath}.zip", "rb") as f:
            st.download_button(
                label="Download JSON Data",
                data=f,
                file_name=f"{uploaded_file_name.replace('.pdf', '')}_results.zip",
                mime="application/zip",
            )
        dp.cleanup()


BACKEND_URL = os.getenv("BACKEND_URL", "http://omnipdf-backend:8003")


def main():
    st.title("OmniPDF: Your PDF Assistant 🦸")
    st.subheader("Upload your PDF file and explore its content")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    if uploaded_file is not None:
        TMP_DATA_DIR = "frontend/data"
        os.makedirs(TMP_DATA_DIR, exist_ok=True)
        file_path = f"{TMP_DATA_DIR}/{uploaded_file.name}"

        # Save file only if not already processed
        if "pdf_file" not in st.session_state or st.session_state.pdf_file != file_path:
            with open(file_path, "wb") as temp_file:
                file_bytes = uploaded_file.read()
                temp_file.write(file_bytes)
                st.session_state.pdf_file = file_path

            # Clear chat history for Chat with Omni
            if "messages" in st.session_state:
                st.session_state.messages = []

            # Define initial variables
            st.session_state.PAGES_DATA = []  # Stores extracted data for each page
            st.session_state.DOCUMENTS = []  # Stores documents for RAG

            # Initialize progress bar
            progress_bar = st.progress(0, "Processing PDF")

            # Retrieve PDF pages
            response = requests.post(
                f"{BACKEND_URL}/pdf_pages",
                files={"file": ("pdf", file_bytes, uploaded_file.type)},
            ).json()
            num_pages = response["num_pages"]

            # Get pages data and documents for each page
            for page_number in range(num_pages):
                response = requests.post(
                    f"{BACKEND_URL}/process_pdf_page",
                    json={"page_number": page_number},
                ).json()

                # Update pages data and documents in session state
                st.session_state.PAGES_DATA.append(response["pages_data"])
                st.session_state.DOCUMENTS.extend(response["documents"])

                # Update progress bar
                prog = (page_number + 1) / num_pages
                if page_number == num_pages:
                    prog = 1
                progress_bar.progress(
                    prog, f"{page_number + 1}/{num_pages} Page Processed"
                )

            st.session_state.ALL_TEXT = "".join(
                [page.get("text", "") for page in st.session_state.PAGES_DATA]
            )
            st.session_state.ALL_TEXT_TRANSLATED = "".join(
                [
                    page.get("translated_text", "")
                    for page in st.session_state.PAGES_DATA
                ]
            )

            # Insert the Documents as embeddings to the vector database
            _ = requests.post(
                f"{BACKEND_URL}/ingest",
                json={"documents": st.session_state.DOCUMENTS},
            )

        original_pdf_column, functionalities_column = st.columns(2)

        with original_pdf_column:
            pdf_display = display_pdf(st.session_state.pdf_file)
            st.session_state.pdf_display = pdf_display
            st.markdown(pdf_display, unsafe_allow_html=True)

        with functionalities_column:
            translate_tab, chat_tab, json_tab = st.tabs(
                ["Translate Text", "Chat with Omni", "Download JSON"]
            )

            with translate_tab:
                vernacular_text = st.text_area(
                    "Enter text to translate to English",
                    # st.session_state.TRANSLATED_TEXT,
                )
                if st.button("Translate Page/Table") and vernacular_text:
                    with st.spinner("Translating text..."):
                        response = requests.post(
                            f"{BACKEND_URL}/translate",
                            json={"text": vernacular_text},
                        ).json()
                    st.markdown(response["translation"])

            with chat_tab:
                # Chat with Omni
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Create a scrollable chat container using st.container()
                chat_container = st.container(height=600)

                num_docs = st.number_input(
                    f"Number of document chunks to fetch (min=1, max={len(st.session_state.DOCUMENTS)}) and improve the resolution of your prompt.",
                    min_value=1,
                    max_value=len(st.session_state.DOCUMENTS),
                    value=min(5, len(st.session_state.DOCUMENTS)),
                    help="Increasing the number of document chunks can lead to more detailed and accurate responses. However, it may also increase processing time and risk truncation due to the model's maximum context length limitations.",
                )

                if prompt := st.chat_input("Ask me anything about your document!"):
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt}
                    )

                    with chat_container:
                        message_placeholder = (
                            st.empty()
                        )  # Placeholder to store messages dynamically

                        # Display chat history inside a container
                        with message_placeholder.container():
                            for message in st.session_state.messages:
                                with st.chat_message(message["role"]):
                                    st.markdown(message["content"])

                        # Create new prompt and get enhanced answer with RAG
                        response = requests.post(
                            f"{BACKEND_URL}/rag_prompt",
                            json={
                                "prompt": prompt,
                                "num_docs": num_docs,
                                "pages_data": st.session_state.PAGES_DATA,
                            },
                        ).json()
                        ans = response["ans"]
                        docs = response["docs"]

                        # Craft the response with citations
                        if len(docs) < num_docs:
                            ans += f"\n\n **References (truncated)**"
                        else:
                            ans += f"\n\n **References**"

                        # Compile citations
                        for doc in docs:
                            # Text documents
                            if doc.get("metadata").get("type") == "text":
                                text_chunk_key = doc.get("metadata").get(
                                    "text_chunk_key"
                                )
                                key_parts = text_chunk_key.split("_")
                                ans += f"\n- [Text Chunk found on page {key_parts[-2]}](#{text_chunk_key})"
                            # Table documents
                            elif doc.get("metadata").get("type") == "table":
                                found = False
                                for page in st.session_state.PAGES_DATA:
                                    for trans_table_summary in page.get(
                                        "translated_tables_summary", []
                                    ):
                                        if trans_table_summary["key"] == doc.get(
                                            "metadata"
                                        ).get("trans_table_summary_key"):
                                            table_key = doc.get("metadata").get(
                                                "trans_table_summary_key"
                                            )
                                            summary = trans_table_summary["summary"]
                                            ans += f"\n- [Table found on page {page.get('page_number')}](#{table_key})"
                                            found = True
                                            break
                                    if found:
                                        break
                            # Image documents
                            elif doc.get("metadata").get("type") == "image":
                                found = False
                                for page in st.session_state.PAGES_DATA:
                                    for image in page.get("images", []):
                                        if image["key"] == doc.get("metadata").get(
                                            "image_caption_key"
                                        ):
                                            img_key = doc.get("metadata").get(
                                                "image_caption_key"
                                            )
                                            caption = image["caption"]
                                            ans += f"\n- [Image found on page {page.get('page_number')}](#{img_key})"
                                            found = True
                                            break
                                    if found:
                                        break

                        # Display answer response
                        st.session_state.messages.append(
                            {"role": "assistant", "content": ans}
                        )

                        with message_placeholder.container():
                            for message in st.session_state.messages:
                                with st.chat_message(message["role"]):
                                    st.markdown(message["content"])

            with json_tab:
                if st.session_state.PAGES_DATA:
                    st.markdown("Download the PDF file in JSON format!")
                    download_pdf_data(
                        uploaded_file.name,
                        st.session_state.PAGES_DATA,
                        st.session_state.ALL_TEXT,
                        st.session_state.ALL_TEXT_TRANSLATED,
                    )

        with st.expander("View Tables"):
            extracted_tables_col, TRANSLATED_TABLES_col = st.columns(2)

            with extracted_tables_col:
                for page in st.session_state.PAGES_DATA:
                    for table_idx, table in enumerate(page.get("tables", [])):
                        st.subheader(
                            f"Table {table_idx + 1} found on page {page.get('page_number')}",
                            anchor=f"table_key_{table_idx + 1}_{page.get('page_number')}",
                        )
                        st.dataframe(table)

            with TRANSLATED_TABLES_col:
                for page in st.session_state.PAGES_DATA:
                    for table_idx, table in enumerate(
                        page.get("translated_tables_summary", [])
                    ):
                        st.subheader(
                            f"Table {table_idx + 1} found on page {page.get('page_number')}",
                            anchor=f"trans_table_key_{page.get('page_number')}_{table_idx + 1}",
                        )
                        st.dataframe(table["translated_table"])

        with st.expander("View WordClouds"):
            maxwords = st.number_input(
                "Top words to display", min_value=1, max_value=1000, value=100
            )
            st.info(f"Displaying top {maxwords} words")

            wordcloud_col, translated_wordcloud_col = st.columns(2)
            with wordcloud_col:
                st.markdown(f"**Vernacular WordCloud**")
                wordcloud = wcg.generate_wordcloud(
                    text=st.session_state.ALL_TEXT,
                    max_words=maxwords,
                    height=200,
                    width=400,
                )
                st.pyplot(wordcloud)
            with translated_wordcloud_col:
                st.markdown(f"**English WordCloud**")
                translated_wordcloud = wcg.generate_wordcloud(
                    text=st.session_state.ALL_TEXT_TRANSLATED,
                    max_words=maxwords,
                    height=200,
                    width=400,
                )
                st.pyplot(translated_wordcloud)

        with st.expander("View Images"):
            image_cols = cycle(st.columns(4))

            for page in st.session_state.PAGES_DATA:
                for image in page.get("images", []):
                    key = image.get("key")
                    caption = image.get("caption")
                    col = next(image_cols)
                    col.subheader(
                        f"Image {key.split('_')[-1]} found on page {key.split('_')[-2]}",
                        anchor=key,
                    )
                    img_bytes = base64.b64decode(image.get("img_b64"))
                    image = Image.open(io.BytesIO(img_bytes))
                    col.image(image, caption=caption, width=360)

        with st.expander("View Text Chunks"):
            for document in st.session_state.DOCUMENTS:
                if document.get("metadata").get("type") == "text":
                    text_chunk_key = (
                        document.get("metadata").get("text_chunk_key").split("_")
                    )
                    st.subheader(
                        f"**Text Chunk {text_chunk_key[-1]} found on page {text_chunk_key[-2]}**",
                        anchor=document.get("metadata").get("text_chunk_key"),
                    )
                    st.markdown(document.get("page_content"))


if __name__ == "__main__":
    main()
