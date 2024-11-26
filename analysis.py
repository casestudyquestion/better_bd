import glob
import os
from dotenv import load_dotenv  

# <---- LLM ----> 
# 1. read files



# Dictionary to store pages from each PDF
pdf_pages = {}

# Iterate through each PDF file in the folder
for file_path in pdf_files:
    try:
        loader = PyPDFLoader(file_path)
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        
        # Use the filename without the directory as the key
        file_key = os.path.basename(file_path)
        pdf_pages[file_key] = pages  # Store pages list as the value for each file
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        pdf_pages[file_path] = f"Error: {str(e)}"  # Optionally store the error message as the value

# 2. clean files 

# 3. parse into LLM 
load_dotenv()  
print(os.getenv("OPENAI_API_KEY"))  

# 4. info extraction 

# 4a. guide to LLM to extract info - flow 1 (company-specific)
prompt1 = ""


# 4b. guide to LLM to extract info - flow 2 (industry-specific)
prompt2 = ""

# <---- streamlit ---->
# 5. flow 1 - show company-specific insights 

# 6. flow 2 - streamlit data analysis


