import base64
import os
import uuid
from difflib import SequenceMatcher as SM

import fitz
import pandas as pd
import pdfplumber
import pytesseract

# import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf2image import convert_from_path
from PIL import Image

from .APIRouter import (
    translate_table,
    translate_text,
    summarize_table,
    summarize_text,
    caption_image,
    CLIENT,
)
from .TableDataProcessor import TableDataProcessor


tbp = TableDataProcessor()


class PDFProcessor:
    """
    A class to process PDFs by extracting images, text, and tables per page.
    """

    def __init__(self, pdf_path, ocr_languages="eng+ara+id+ms"):
        """
        Initializes the PDFProcessor.

        Args:
            pdf_path (str): Path to the PDF file.
            ocr_languages (str, optional): Languages for OCR (default: "eng+ara+id+ms").
        """
        self.pdf_path = pdf_path
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        try:
            self.pdf = pdfplumber.open(self.pdf_path)
        except Exception as e:
            print(f"Error opening PDF file as pdfplumber: {e}")
        try:
            self.pdf_for_images = fitz.open(self.pdf_path)
        except Exception as e:
            print(f"Error opening PDF file as pdfplumber: {e}")
        self.ocr_languages = ocr_languages
        self.pages_data = []  # Stores extracted data for each page
        self.documents = []  # Stores documents for RAG
        self._initial_cleanup()
        # self._process_pdf()
        # self._extract_images()  # Extract images after processing text and tables

    def get_pages(self):
        return len(self.pdf.pages)

    def close_pdf(self):
        if self.pdf:
            self.pdf.close()

    def process_pdf_page(self, page_number: int) -> tuple[dict, list]:
        """Extracts pages data and documents.

        Both iterables contain images, text (excluding table text), and tables
        for each page.
        """

        # Store pages data and documents
        pages_data = {}
        documents = []

        # Get page content
        page_content = self.pdf.pages[page_number]

        # Extract tables
        tables = self._extract_tables(page_content)
        # Convert PDF page into PNG for OCR purposes
        page_images = self._convert_page_to_images(page_number)
        # Extract text from images (OCR)
        raw_text = self._extract_text_from_images(page_images)
        if raw_text and tables:
            # Remove table text from the extracted text
            filtered_text = self._remove_fuzzy_match(tables=tables, raw_text=raw_text)
        else:
            filtered_text = raw_text

        # Split the filtered text into chunks for better translation
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)
        text_chunks = text_splitter.split_text(filtered_text)

        translated_text = ""
        for chunk_idx, text_chunk in enumerate(text_chunks):
            translated_text_chunk = translate_text(text_chunk, CLIENT)
            translated_text += translated_text_chunk

            # Add text documents
            documents.append(
                {
                    "page_content": translated_text_chunk,
                    "metadata": {
                        "text_chunk_key": f"text_chunk_{page_number + 1}_{chunk_idx + 1}",
                        "type": "text",
                    },
                }
            )

        # Translate tables and summarize translated tables
        translated_tables_summary = []
        for table_idx, table in enumerate(tables):
            key = f"trans_table_summary_{page_number + 1}_{table_idx + 1}"
            translated_table = translate_table(tbp.format_for_json(table), CLIENT)
            summary = summarize_table(translated_table, CLIENT)
            translated_tables_summary.append(
                {
                    "key": key,
                    "translated_table": translated_table,
                    "summary": summary,
                }
            )

            # Add table documents
            documents.append(
                {
                    "page_content": summary,
                    "metadata": {
                        "trans_table_summary_key": key,
                        "type": "table",
                    },
                }
            )

        # Extract images
        images, image_documents = self._extract_images(page_number)

        # Add image documents
        documents.extend(image_documents)

        # Store extracted data
        pages_data = {
            "page_number": page_number + 1,
            "text": filtered_text,
            "translated_text": translated_text,
            "tables": tables,
            "translated_tables_summary": translated_tables_summary,
            "images": images,
        }

        # Clean up temporary images
        self._cleanup_images(page_images)

        # Close pdf at the end
        if page_number == self.get_pages():
            self.close_pdf()

        return pages_data, documents

    def _process_pdf(self):
        """Extracts images, text (excluding table text), and tables from each page."""

        with pdfplumber.open(self.pdf_path) as pdf:
            # progress_bar = st.progress(0, "Processing PDF")  # Initialize progress bar
            for page_number, page in enumerate(pdf.pages):
                # Extract tables from the page
                tables = self._extract_tables(page)
                # Convert PDF page into PNG for OCR purposes
                page_images = self._convert_page_to_images(page_number)
                # Extract text from images (OCR)
                raw_text = self._extract_text_from_images(page_images)
                if raw_text and tables:
                    # Remove table text from the extracted text
                    filtered_text = self._remove_fuzzy_match(
                        tables=tables, raw_text=raw_text
                    )
                else:
                    filtered_text = raw_text

                # Split the filtered text into chunks for better translation
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024)
                text_chunks = text_splitter.split_text(filtered_text)

                translated_text = ""
                for chunk_idx, text_chunk in enumerate(text_chunks):
                    translated_text_chunk = translate_text(text_chunk, CLIENT)
                    translated_text += translated_text_chunk

                    # Add text documents
                    self.documents.append(
                        {
                            "page_content": translated_text_chunk,
                            "metadata": {
                                "text_chunk_key": f"text_chunk_{page_number + 1}_{chunk_idx + 1}",
                                "type": "text",
                            },
                        }
                    )

                # Translate tables and summarize translated tables
                translated_tables_summary = []
                for table_idx, table in enumerate(tables):
                    key = f"trans_table_summary_{page_number + 1}_{table_idx + 1}"
                    translated_table = translate_table(
                        tbp.format_for_json(table), CLIENT
                    )
                    summary = summarize_table(translated_table, CLIENT)
                    translated_tables_summary.append(
                        {
                            "key": key,
                            "translated_table": translated_table,
                            "summary": summary,
                        }
                    )

                    # Add table documents
                    self.documents.append(
                        {
                            "page_content": summary,
                            "metadata": {
                                "trans_table_summary_key": key,
                                "type": "table",
                            },
                        }
                    )

                # Store extracted data
                self.pages_data.append(
                    {
                        "page_number": page_number + 1,
                        "text": filtered_text,
                        "translated_text": translated_text,
                        "tables": tables,
                        "translated_tables_summary": translated_tables_summary,
                        "images": [],
                    }
                )

                # Clean up temporary images
                self._cleanup_images(page_images)

                # Update progress bar
                prog = (page_number + 1) / len(pdf.pages)
                if page_number == len(pdf.pages):
                    prog = 1

                # progress_bar.progress(
                #     prog, f"{page_number + 1}/{len(pdf.pages)} Page Processed"
                # )

    def _extract_tables(self, page) -> list:
        """Extracts tables as structured data."""
        return [pd.DataFrame(table).values.tolist() for table in page.extract_tables()]

    def _convert_page_to_images(self, page_number: int) -> list:
        """Converts a PDF page to an image using pdf2image."""
        img_dir = "backend/ocr_pdf"
        os.makedirs(img_dir, exist_ok=True)

        page_images = []
        pages = convert_from_path(
            self.pdf_path, first_page=page_number + 1, last_page=page_number + 1
        )

        for idx, img in enumerate(pages):
            img_path = os.path.join(
                img_dir, f"temp_page_{page_number + 1}_{idx + 1}.png"
            )
            img.save(img_path, "PNG")
            page_images.append(img_path)

        return page_images

    def _extract_images(self, page_number: int) -> tuple[list, list]:
        """Extracts embedded images from a PDF page and saves them as PNGs."""

        img_dir = "backend/extracted_images"
        os.makedirs(img_dir, exist_ok=True)

        # Store pages data and documents
        images = []
        image_documents = []

        # Extract images
        page = self.pdf_for_images[page_number]
        for img_index, img_obj in enumerate(page.get_images(full=True)):
            xref = img_obj[0]
            base_image = self.pdf_for_images.extract_image(xref)
            img_bytes = base_image["image"]
            img_filename = f"embedded_page_{page_number + 1}_{img_index + 1}.png"
            img_path = f"{img_dir}/{img_filename}"
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            print(f"✅ Successfully extracted: {img_path}")

            key = f"image_caption_{page_number + 1}_{img_index + 1}"
            caption = caption_image(img_path, CLIENT)
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            # Add image documents
            image_documents.append(
                {
                    "page_content": caption,
                    "metadata": {
                        "image_caption_key": key,
                        "type": "image",
                    },
                }
            )

            # Store image data in pages_data
            images.append(
                {
                    "key": key,
                    "img_filename": img_filename,
                    "image_url": img_path,
                    "img_b64": img_b64,
                    "caption": caption,
                }
            )

        # Close the pdf for images at the end
        if page_number == len(self.pdf_for_images):
            self.pdf_for_images.close()

        return images, image_documents

    # def _extract_images(self) -> None:
    #     """Extracts embedded images from a PDF page and saves them as PNGs."""
    #     img_dir = "backend/extracted_images"
    #     os.makedirs(img_dir, exist_ok=True)
    #     doc = fitz.open(self.pdf_path)
    #     # progress_bar = st.progress(0, "Processing Images")  # Initialize progress bar
    #     for page_number in range(len(doc)):
    #         page = doc[page_number]
    #         for img_index, img_obj in enumerate(page.get_images(full=True)):
    #             xref = img_obj[0]
    #             base_image = doc.extract_image(xref)
    #             img_bytes = base_image["image"]
    #             img_path = (
    #                 f"{img_dir}/embedded_page_{page_number + 1}_{img_index + 1}.png"
    #             )
    #             with open(img_path, "wb") as f:
    #                 f.write(img_bytes)
    #             print(f"✅ Successfully extracted: {img_path}")

    #             key = f"image_caption_{page_number + 1}_{img_index + 1}"
    #             caption = caption_image(img_path, CLIENT)

    #             # Add image documents
    #             self.documents.append(
    #                 {
    #                     "page_content": caption,
    #                     "metadata": {
    #                         "image_caption_key": key,
    #                         "type": "image",
    #                     },
    #                 }
    #             )

    #             # Store image data in pages_data
    #             self.pages_data[page_number]["images"].append(
    #                 {
    #                     "key": key,
    #                     "image_url": img_path,
    #                     "image_bytes": img_bytes,
    #                     "caption": caption,
    #                 }
    #             )

    #         # Update progress bar
    #         prog = (page_number + 1) / len(doc)
    #         if page_number == len(doc):
    #             prog = 1

    # progress_bar.progress(
    #     prog, f"{page_number + 1}/{len(doc)} Page Processed (Images)"
    # )

    def _extract_text_from_images(self, images: list) -> str:
        """Extracts text from images using OCR (supports Arabic and multiple languages)."""
        return "\n".join(
            pytesseract.image_to_string(
                Image.open(img), lang=self.ocr_languages
            ).strip()
            for img in images
        ).strip()

    def _remove_fuzzy_match(self, tables: list, raw_text: str) -> list:
        """Removes text that matches table content using fuzzy matching."""
        # Split the raw text into rows based on new line characters
        rows_of_text = raw_text.split("\n")

        # Initialize lists to store cleaned and filtered text and table rows
        cleaned_text_rows = []
        filtered_text_rows = []
        joined_table = []

        # Clean up the text rows
        for row in rows_of_text:
            cleaned_text_rows.append(row.replace("|", ""))

        # Convert the tables into a single list of strings
        for row in tables[0]:
            row = filter(None, row)
            joined_table.append(" ".join(row))

        # Find text rows that match table content using fuzzy matching
        for clean_row in cleaned_text_rows:
            for table_row in joined_table:
                if SM(None, clean_row, table_row).ratio() > 0.7:
                    filtered_text_rows.append(clean_row)
                    break
        # Return the cleaned text rows that do not match any table content
        return "\n".join(
            [item for item in cleaned_text_rows if item not in filtered_text_rows]
        )

    def _initial_cleanup(self) -> None:
        """Initial cleanup of temporary files."""

        # Set the directory paths for temporary files
        embedded_dir = r"backend/extracted_images/"
        extracted_dir = r"backend/ocr_pdf/"
        pdf_dir = r"backend/data/"

        # Remove any existing temporary files prior to processing
        for dir in [embedded_dir, extracted_dir]:
            if os.path.exists(dir):
                for file in os.listdir(dir):
                    file_path = os.path.join(dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

    def _cleanup_images(self, images: list) -> None:
        """Removes temporary image files after processing."""
        for img_path in images:
            if os.path.exists(img_path):
                os.remove(img_path)

    def get_page_data(self, page_number: int) -> dict:
        """
        Retrieves extracted data for a specific page.

        Args:
            page_number (int): Page number (1-based index).

        Returns:
            dict: Extracted text, tables, and images for the page.
        """
        if 1 <= page_number <= len(self.pages_data):
            return self.pages_data[page_number - 1]
        raise IndexError("Page number out of range.")

    def get_all_data(self) -> list[dict]:
        """
        Retrieves extracted data for the entire PDF.

        Returns:
            list: List of dictionaries containing text, tables, and images for each page.
        """
        return self.pages_data

    def get_all_documents(self) -> list[dict]:
        """
        Retrieves extracted documents for the entire PDF.

        Returns:
            list: List of Documents containing text, tables, and images for each page.
        """
        return self.documents
