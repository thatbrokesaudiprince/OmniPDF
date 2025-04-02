import os
import pdfplumber
import streamlit as st
import fitz
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
from difflib import SequenceMatcher as SM


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
        self.ocr_languages = ocr_languages
        self.pages_data = []  # Stores extracted data for each page
        self._initial_cleanup()
        self._process_pdf()
        self._extract_images()  # Extract images after processing text and tables

    def _process_pdf(self):
        """Extracts images, text (excluding table text), and tables from each page."""

        with pdfplumber.open(self.pdf_path) as pdf:
            progress_bar = st.progress(0, "Processing PDF")  # Initialize progress bar
            for i, page in enumerate(pdf.pages):
                # Extract tables from the page
                tables = self._extract_tables(page)
                # Convert PDF page into PNG for OCR purposes
                page_images = self._convert_page_to_images(i)
                # Extract text from images (OCR)
                raw_text = self._extract_text_from_images(page_images)
                if raw_text and tables:
                    # Remove table text from the extracted text
                    filtered_text = self._remove_fuzzy_match(
                        tables=tables, raw_text=raw_text
                    )
                else:
                    filtered_text = raw_text
                # Store extracted data
                self.pages_data.append(
                    {
                        "page_number": i + 1,
                        "text": filtered_text,
                        "tables": tables,
                        "images": [],
                    }
                )

                # Clean up temporary images
                self._cleanup_images(page_images)
                # Update progress bar
                prog = (i + 1) / len(pdf.pages)
                if i == len(pdf.pages):
                    prog = 1

                progress_bar.progress(prog, f"{i+1}/{len(pdf.pages)} Page Processed")

    def _extract_tables(self, page) -> list:
        """Extracts tables as structured data."""
        return [pd.DataFrame(table).values.tolist() for table in page.extract_tables()]

    def _convert_page_to_images(self, page_number: int) -> list:
        """Converts a PDF page to an image using pdf2image."""
        img_dir = "ocr_pdf"
        os.makedirs(img_dir, exist_ok=True)

        page_images = []
        pages = convert_from_path(
            self.pdf_path, first_page=page_number + 1, last_page=page_number + 1
        )

        for idx, img in enumerate(pages):
            img_path = os.path.join(img_dir, f"temp_page_{page_number + 1}_{idx}.png")
            img.save(img_path, "PNG")
            page_images.append(img_path)

        return page_images

    def _extract_images(self) -> None:
        """Extracts embedded images from a PDF page and saves them as PNGs."""
        img_dir = "extracted_images"
        os.makedirs(img_dir, exist_ok=True)
        doc = fitz.open(self.pdf_path)
        for page_number in range(len(doc)):
            page = doc[page_number]
            for img_index, img_obj in enumerate(page.get_images(full=True)):
                xref = img_obj[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_path = (
                    f"extracted_images/embedded_page_{page_number + 1}_{img_index}.png"
                )
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                print(f"✅ Successfully extracted: {img_path}")
                # Store the image path in the page data
                self.pages_data[page_number]["images"].append(img_path)

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
        embedded_dir = r"extracted_images/"
        extracted_dir = r"ocr_pdf/"
        pdf_dir = r"data/"

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
