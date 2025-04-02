import streamlit as st
import os
from utils import *  # Import functions from utils.py
from api_utils import *  # Import functions from api_utils.py
import json
from classes.WordCloudGenerator import WordCloudGenerator
from classes.PDFProcessor import PDFProcessor
from classes.TableDataProcessor import TableDataProcessor

st.set_page_config(layout="wide", page_title="OmniPDF")

wcg = WordCloudGenerator()
tbp = TableDataProcessor()


def main():

    # PDF upload and processing
    st.title("PDF Summarization App")
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        file_path = f"data/{uploaded_file.name}"

        # Save uploaded file
        with open(file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
            st.info("Uploaded PDF File")

        # Extract text and tables from the PDF
        pdf = PDFProcessor(file_path)

        pages = (
            pdf.get_all_data()
        )  # list of dictionaries representing respective page text, images and tables

        ALL_TEXT = ""
        for page in pages:
            ALL_TEXT += page.get("text")

        translated_tables = []
        translated_text = []

        # Display PDF and extracted content in a two-column layout
        original_pdf_column, functionalities_column = st.columns(2)

        with original_pdf_column:
            pdf_display = display_pdf(file_path)  # Display uploaded PDF
            st.markdown(pdf_display, unsafe_allow_html=True)

        with functionalities_column:

            translate_tab, summary_tab, chat_tab = st.tabs(
                ["Translation text to English", "View Summary", "Chat with Omni"]
            )

            with translate_tab:

                vernacular_text = st.text_area("Enter text to translate to english")
                if st.button("Translate Page/Table") and vernacular_text:

                    with st.spinner("Translating text into English"):

                        translated_text = translate_text_api(vernacular_text)[
                            "translation"
                        ]
                    st.markdown(translated_text)

            with summary_tab:
                if st.button("Summarize"):

                    return

            with chat_tab:
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if prompt := st.chat_input("Ask me anything your document!"):
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt}
                    )

                response = f"Echo: {prompt}"
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

        with st.expander("View Tables"):
            extracted_tables_col, translated_tables_col = st.columns(2)

            with extracted_tables_col:
                st.info("Extracted Tables")
                for page in pages:
                    temp = page.get("tables")
                    if temp:
                        for table in temp:
                            st.write(f"Table on Page {page.get('page_number')}")
                            st.dataframe(table)

            with translated_tables_col:
                if st.button("Translate Tables"):
                    for page in pages:
                        temp = page.get("tables")
                        if temp:
                            for table in temp:
                                st.write(f"Table on Page {page.get('page_number')}")
                                translated_table = translate_table_api(
                                    tbp.format_for_json(table)
                                )["translated_table"]
                                st.dataframe(translated_table)
                                translated_tables.append(translated_table)

        with st.expander("View WordCloud"):
            maxwords = st.number_input(
                "Top words to display", min_value=1, max_value=1000
            )
            if maxwords > 1:
                st.info(f"Displaying top {maxwords} words")
                wordcloud = wcg.generate_wordcloud(
                    text=ALL_TEXT, max_words=int(maxwords), height=200, width=400
                )
                st.pyplot(wordcloud, use_container_width=True)

        with st.expander("View Images"):
            for page in pages:
                images = page.get("images")
                if images:
                    for img in images:
                        st.image(img, caption=f"Page {page.get('page_number')}")


if __name__ == "__main__":
    main()
