import pdfplumber
import streamlit as st
import base64
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


# @st.cache_data
def display_pdf(file):
    """Display PDF in an iframe on Streamlit."""
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    return pdf_display


def extract_pdf_text_tables(file):
    """Extract text and tables from a PDF."""
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
        tables = [page.extract_tables() for page in pdf.pages]
    return pages, tables
