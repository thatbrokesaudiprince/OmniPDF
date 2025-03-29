import requests
import streamlit as st
import json

API_BASE_URL = "http://localhost:5000/api/"  # Your FastAPI API base URL

def translate_table_api(table_data):
    """
    Translates a table using the /api/translate-table endpoint.

    Args:
        table_data (list): Nested list representing the table to translate.

    Returns:
        dict: The API response as a dictionary, or None if an error occurs.
    """
    url = f"{API_BASE_URL}translate-table"
    data = {"table_data": table_data}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error translating table: {e}")
        return None

def translate_text_api(text_prompt):
    """
    Translates text using the /api/translate-text endpoint.

    Args:
        text_prompt (str): The text to translate.

    Returns:
        dict: The API response as a dictionary, or None if an error occurs.
    """
    url = f"{API_BASE_URL}translate-text"
    data = {"prompt": text_prompt}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error translating text: {e}")
        return None
    
# print(translate_text_api("Setiap hari saya pergi ke sekolah")['translation'])
# print(translate_table_api([['No', 'Komando / Unsur', 'Nama Panggilan', None], [None, None, 'Telefoni', 'Telegrafi'], ['1.', 'Panglima Kogasgabfib', 'Suleman', 'R8L'], ['2.', 'Komandan Pasrat', 'Mirah', 'M0H'], ['3.', 'Seluruh Unsur BTK', 'Pantar-A', 'P1R'], ['4.', 'Dan ST BTK/ KRI PKR 1', 'Pantar', 'P0R'], ['5.', 'KRI PKR 1', 'Pantar-1', 'P2R'], ['6.', 'KRI PKR 1', 'Pantar-2', 'P3R'], ['7.', 'Heli HS-4201', 'Pantar-3', 'P4R'], ['8.', 'PKBT Kogasfib/PKSB', 'Pantar-4', 'P5R'], ['9.', 'PKBT Pasrat', 'Pantar-5', 'P6R'], ['10.', 'Pabung BTK', 'Pantar-6', 'P7R'], ['11.', 'Pabung BTA', 'Pantar-7', 'P8R'], ['12.', 'Pabung BTU/SUL', 'Pantar-8', 'P9R'], ['13.', 'PD BTK', 'Pantar-9', 'P1T'], ['14.', 'PD BTA', 'Pantar-10', 'P2T'], ['15.', 'PD BTU/SUL', 'Pantar-11', 'P3T']]))