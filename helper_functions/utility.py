import streamlit as st
import random
import hmac
import numpy as np
import requests
import PyPDF2
import os
import re
from dotenv import load_dotenv
import openai
from openai import OpenAI
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# """  
# this file includes key functions used in main.py 
# """  

# function for streamlit checking password 
def check_password():  
    """Returns `True` if the user had the correct password."""  
    def password_entered():  
        """Checks whether a password entered by the user is correct."""  
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):  
            st.session_state["password_correct"] = True  
            del st.session_state["password"]  # Don't store the password.  
        else:  
            st.session_state["password_correct"] = False  
    # Return True if the passward is validated.  
    if st.session_state.get("password_correct", False):  
        return True  
    # Show input for password.  
    st.text_input(  
        "Password", type="password", on_change=password_entered, key="password"  
    )  
    if "password_correct" in st.session_state:  
        st.error("😕 Password incorrect")  
    return False

def download_pdf_to_memory(pdf_url):
    """
    Downloads a PDF file from the given URL and stores it in memory.
    """
    response = requests.get(pdf_url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception(f"Failed to download the file. HTTP Status: {response.status_code}")

def extract_text_from_pdf(file_like_object, exclude_section=None):
    """
    Extracts text from a PDF file in memory. Excludes less relevant sections.
    """
    exclude_sections = ["Message to Shareholders", "Board of Directors"]

    if exclude_sections is not None:
        exclude_sections.append(exclude_section)

    # Compile a regex pattern to detect excluded sections
    exclude_pattern = re.compile("|".join(re.escape(section) for section in exclude_sections), re.IGNORECASE)
    
    reader = PdfReader(file_like_object)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text()
    return extracted_text

def semantic_chunking(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits the text into semantically meaningful chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def analyse_with_gpt(chunks, prompt_template, model="gpt-4o-mini"):
    
    results = []
    for chunk in chunks:
        prompt = prompt_template.format(chunk=chunk)

        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0, # lowest randomness
        )
        results.append(response)

    return response.choices[0].message.content

def estimate_tokens(prompt, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))