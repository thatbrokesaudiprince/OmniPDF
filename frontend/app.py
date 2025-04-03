import streamlit as st
import os
from utils import *  # Import functions from utils.py
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from classes.WordCloudGenerator import WordCloudGenerator
from classes.PDFProcessor import PDFProcessor
from classes.TableDataProcessor import TableDataProcessor
from classes.RAGHelper import RAGHelper
from classes.APIRouter import (
    translate_text,
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
            # Clear chat history for Chat with Omni
            if "messages" in st.session_state:
                st.session_state.messages = []

            with open(file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())
                st.session_state.pdf_file = file_path

            pdf = PDFProcessor(file_path)
            st.session_state.PAGES_DATA = pdf.get_all_data()
            st.session_state.ALL_TEXT = "".join(
                page.get("text", "") for page in st.session_state.PAGES_DATA
            )
            st.session_state.ALL_TEXT_TRANSLATED = "".join(
                page.get("translated_text", "") for page in st.session_state.PAGES_DATA
            )

            # Split text into chunks of 1024 for RAG
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)
            text_docs = text_splitter.split_text(st.session_state.ALL_TEXT_TRANSLATED)

            # Compile the documents
            st.session_state.DOCUMENTS = []

            # Text documents
            for i, doc in enumerate(text_docs):
                st.session_state.DOCUMENTS.append(
                    Document(
                        page_content=doc,
                        metadata={
                            "text_chunk_key": f"text_chunk{str(i + 1)}",
                            "source": uploaded_file.name,
                            "type": "text",
                        },
                    )
                )

            # Table documents
            for page in st.session_state.PAGES_DATA:
                for trans_table_summary in page.get("translated_tables_summary", []):
                    key = trans_table_summary["key"]
                    summary = trans_table_summary["summary"]

                    st.session_state.DOCUMENTS.append(
                        Document(
                            page_content=summary,
                            metadata={
                                "trans_table_summary_key": key,
                                "source": uploaded_file.name,
                                "type": "table",
                            },
                        )
                    )

            # Image documents
            for page in st.session_state.PAGES_DATA:
                for image in page.get("images", []):
                    print(image)
                    key = image["key"]
                    caption = image["caption"]

                    st.session_state.DOCUMENTS.append(
                        Document(
                            page_content=caption,
                            metadata={
                                "image_caption_key": key,
                                "source": uploaded_file.name,
                                "type": "image",
                            },
                        )
                    )

            # Insert the Documents as embeddings to the vector database
            rh.add_docs_to_chromadb(st.session_state.DOCUMENTS)

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
                # Chat with Omni
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                # Create a scrollable chat container using st.container()
                chat_container = st.container(height=600)

                num_docs = st.number_input(
                    "Documents to fetch",
                    min_value=1,
                    max_value=len(st.session_state.DOCUMENTS),
                    value=min(5, len(st.session_state.DOCUMENTS) // 2),
                    help="Specify the number of relevant documents to fetch and improve your prompt. Please note that the number may be limited and truncated by the model's max context length. The more the number of documents specified, the longer the process is.",
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

                        # Retrieve top-k documents from vector database relevant to the prompt
                        rel_docs = rh.retrieve_relevant_docs(prompt, num_docs)

                        # Create new prompt with relevant documents
                        response, docs = rag_prompt(
                            prompt, rel_docs, st.session_state.PAGES_DATA, CLIENT
                        )

                        # Craft the response with citations
                        if len(docs) < num_docs:
                            response += f"\n\n **References (truncated)**"
                        else:
                            response += f"\n\n **References**"

                        # Compile citations
                        for doc in docs:
                            # Text documents
                            if doc.metadata.get("type") == "text":
                                response += f"\n- Text Chunk {doc.metadata.get('text_chunk_key').replace('text_chunk', '')}"
                            # Table documents
                            elif doc.metadata.get("type") == "table":
                                found = False
                                for page in st.session_state.PAGES_DATA:
                                    for trans_table_summary in page.get(
                                        "translated_tables_summary", []
                                    ):
                                        if trans_table_summary[
                                            "key"
                                        ] == doc.metadata.get(
                                            "trans_table_summary_key"
                                        ):
                                            summary = trans_table_summary["summary"]
                                            response += f"\n- Table on Page: {page.get('page_number')}"
                                            found = True
                                            break
                                    if found:
                                        break
                            # Image documents
                            elif doc.metadata.get("type") == "image":
                                found = False
                                for page in st.session_state.PAGES_DATA:
                                    for image in page.get("images", []):
                                        if image["key"] == doc.metadata.get(
                                            "image_caption_key"
                                        ):
                                            caption = image["caption"]
                                            response += f"\n- Image on Page: {page.get('page_number')}"
                                            found = True
                                            break
                                    if found:
                                        break

                        # Display response
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response}
                        )

                        with message_placeholder.container():
                            for message in st.session_state.messages:
                                with st.chat_message(message["role"]):
                                    st.markdown(message["content"])

        with st.expander("View Tables"):
            extracted_tables_col, TRANSLATED_TABLES_col = st.columns(2)

            with extracted_tables_col:
                for page in st.session_state.PAGES_DATA:
                    for table in page.get("tables", []):
                        st.write(f"Table on Page {page.get('page_number')}")
                        st.dataframe(table)

            with TRANSLATED_TABLES_col:
                for page in st.session_state.PAGES_DATA:
                    for table in page.get("translated_tables_summary", []):
                        st.write(f"Table on Page {page.get('page_number')}")
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
            for page in st.session_state.PAGES_DATA:
                for image in page.get("images", []):
                    caption = image["caption"]
                    st.image(
                        image["image_url"],
                        caption=f"{caption}\nFound on Page: {page.get('page_number')}",
                    )

        with st.expander("View Text Chunks"):
            for document in st.session_state.DOCUMENTS:
                if document.metadata.get("type") == "text":
                    st.markdown(
                        f"**Text Chunk {document.metadata.get('text_chunk_key').replace('text_chunk', '')}**"
                    )
                    st.markdown(document.page_content)

        # print(st.session_state.PAGES_DATA)


if __name__ == "__main__":
    main()
