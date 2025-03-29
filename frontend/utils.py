from transformers import AutoModelForCausalLM, AutoTokenizer
import pdfplumber
import streamlit as st
import base64

def process_table_data(table):
    """
    Processes table data for JSON serialization, handling special cases.

    Args:
        table: A nested list representing the table data.

    Returns:
        A processed nested list suitable for JSON conversion.
    """
    processed_table = []
    for row in table:
        processed_row = []
        for cell in row:
            if isinstance(cell, str):
                # Replace newlines and single quotes, and handle "null" string.
                processed_cell = cell.replace('\n', '\\n').replace("'", '"')
                if processed_cell == "null":
                    processed_cell = None
            elif cell is None:
                processed_cell = None  # Use None directly for JSON null
            else:
                processed_cell = cell  # Keep other types as is

            processed_row.append(processed_cell)
        processed_table.append(processed_row)
    return processed_table

@st.cache_data
def display_pdf(file):
    """Display PDF in an iframe on Streamlit."""
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

@st.cache_data
def extract_pdf_text_tables(file):
    """Extract text and tables from a PDF."""
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
        tables = [page.extract_tables() for page in pdf.pages]
    return pages, tables
