import streamlit as st
import pandas as pd
import numpy as np
import openai
from openai import OpenAI
import os
import re
import requests
import PyPDF2
import langchain
import langchain_community
from langchain_community.document_loaders import PyPDFLoader
import asyncio
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import tiktoken
from PIL import Image
from dotenv import load_dotenv
from helper_functions.utility import (
    check_password,
    extract_text_from_pdf,
    download_pdf_to_memory,
    extract_text_from_pdf,
    semantic_chunking,
    analyse_with_gpt,
    estimate_tokens
)

# load environment variables from .env file
load_dotenv()

# set OpenAI API key from the environment variable and pass to OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai.api_key)


# Initialize streamlit session state to store results and URLs
if 'results' not in st.session_state:
    st.session_state.results = None
if 'urls' not in st.session_state:
    st.session_state.urls = []

# region <---- Streamlit App ----->
st.title("Better BD")

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main", "About Us", "Methodology"])

# expander 
with st.expander("Click for more information"):
    st.write("IMPORTANT NOTICE: This web application is developed as a proof-of-concept prototype. The information provided here is NOT intended for actual usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters. Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output. Always consult with qualified professionals for accurate and personalized advice.")

# Page: Main
if page == "Main":

    st.subheader("Main")

    # Input for URLs
    url_input = st.text_input(
        "Enter PDF URL to the relevant annual report e.g., https://www.first-resources.com/wp-content/uploads/2024/08/20240403233925_91270.pdf",
        value='; '.join(st.session_state.get('urls', []))
    )
    submit = st.button("Submit")

    # Validate URLs
    if submit:
        try:
            # Step 1: Process multiple URLs
            urls = [url.strip() for url in url_input.split(";")]
            combined_text = ""

            # Step 2: Download and extract text from each PDF
            for url in urls:
                st.write(f"Processing PDF {url}, please hold on. This may take some time...")
                pdf_file = download_pdf_to_memory(url)
                text = extract_text_from_pdf(pdf_file)
                combined_text += text

            st.write("Text extraction completed.")

            # Step 3: Estimate tokens for the combined text
            total_tokens = estimate_tokens(combined_text, model="gpt-4o-mini")
            st.write(f"Estimated tokens for the document: {total_tokens}")

            # Stop processing if tokens exceed a limit
            if total_tokens > 4000:  # Adjust limit as per model
                st.error("This document is too large. Please reduce its size.")
                st.stop()

            # Step 4: Perform semantic chunking
            st.write("Performing semantic chunking...")
            chunks = semantic_chunking(combined_text)
            st.write(f"Text split into {len(chunks)} chunks.")

            # Step 5: Prep prompts
            swot_prompt_template = PromptTemplate(
                input_variables=["chunk"],
                template="You are a consultant and professional business analyst. Based on the annual report provided, including information such as the company's plans and strategic priorities, conduct a SWOT analysis of the company: {chunk}"
            )
            financial_prompt_template = PromptTemplate(
                input_variables=["chunk"],
                template="You are a professional financial consultant. Extract relevant and key financial information from the report, and assess the company's financial health. Highlight any notable details (good or bad): {chunk}"
            )

            # Step 5: Use GPT-4o-mini 
            st.write("Running SWOT analysis...")
            swot_results = analyse_with_gpt(chunks, swot_prompt_template)

            st.write("Analysing financial health...")
            financial_results = analyse_with_gpt(chunks, financial_prompt_template)

            # Step 6: Display results
            st.subheader("SWOT Analysis Results")
            for idx, result in enumerate(swot_results, 1):
                st.markdown(f"**Chunk {idx}:** {result}")

            st.subheader("Financial Assessment Results")
            for idx, result in enumerate(financial_results, 1):
                st.markdown(f"**Chunk {idx}:** {result}")

        except Exception as e:
            st.error(f"Error: {e}")


# "About Us" page - A detailed page outlining the project scope, objectives, data sources, and features.
elif page == "About Us":
    st.subheader("About Us")
    st.image("/Users/siqingchong/Documents/better_bd/project_scope/Slide4.png", caption="Problem Statement: Problem") 
    st.image("/Users/siqingchong/Documents/better_bd/project_scope/Slide5.png", caption="Problem Statement: Urgency & Severity") 
    st.image("/Users/siqingchong/Documents/better_bd/project_scope/Slide6.png", caption="Proposed Solution") 
    st.image("/Users/siqingchong/Documents/better_bd/project_scope/Slide7.png", caption="Impact1") 
    st.image("/Users/siqingchong/Documents/better_bd/project_scope/Slide8.png", caption="Impact2") 
    st.write("A later-stage enhancement is to store the insights generate per company queried, so that companies can be benchmarked against one another (SWOT and financial strength).")
    st.write("Current app supports multiple pdf urls but requires a significant longer loading time. Hence, removed from instructions.")

# "Methodology" page - A comprehensive explanation of the data flows and implementation details.
elif page == "Methodology":
    st.subheader("Methodology")
    st.write("Better BD (Business Development) serves to help EnterpriseSG officers in industry clusters better understand, and thus help their accounts and companies.")
    st.write("There are two main functions to Better BD:")
    st.write("First, on a company-level, synthesise and extract the company's quick 'stats' and upcoming plans from their annual report (if publicly listed).")
    st.write("Second, based on the financial information provided, establish the industry benchmark and where each company stand in relation to others.")
    st.image("methodology_flow.png", caption="Methodology Overview")  

# end region <---- Streamlit App  ----->
