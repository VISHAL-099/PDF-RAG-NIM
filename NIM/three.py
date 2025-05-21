import streamlit as st
import os
import pandas as pd
import time
from datetime import datetime
import pdfplumber
import shutil
import faiss
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set NVIDIA API key
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Initialize NVIDIA LLM
llm = ChatNVIDIA(model="meta/llama-3.1-70b-instruct")

# Streamlit app title
st.title("NVIDIA NIM Demo: Document Retrieval and Querying")

# Folder to store saved data
SAVED_DATA_DIR = "saved_data"
os.makedirs(SAVED_DATA_DIR, exist_ok=True)

# Function to extract text from PDFs using pdfplumber and save to CSV
def extract_text_to_csv(pdf_files):
    data = []
    csv_file = None
    try:
        for pdf_file in pdf_files:
            # Using pdfplumber to extract text
            with pdfplumber.open(pdf_file) as pdf:
                st.write(f"Extracting content from {pdf_file}...")
                documents = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        documents.append(text)
                    else:
                        st.warning(f"Warning: No extractable text found on page {page.page_number} of {pdf_file}.")
                
                if not documents:
                    st.warning(f"No extractable content found in {pdf_file}. Skipping CSV creation.")
                    continue  # Skip to next file if no content was found

                # Flatten the list of document contents and create CSV data
                content = "\n".join(documents)
                data.append({"Document Name": os.path.basename(pdf_file), "Content": content})

            # Check if we have data to save
            if not data:
                st.warning(f"No content extracted from {pdf_file}. Skipping CSV creation.")
                continue

            # Generate a unique file name with a timestamp for each directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = os.path.join(SAVED_DATA_DIR, f"extracted_text_{timestamp}_{os.path.basename(pdf_file)}.csv")

            # Save extracted data to a CSV
            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)
            st.success(f"Extracted text saved to {csv_file}")
        
        return csv_file  # Return the name of the generated CSV

    except Exception as e:
        st.error(f"Error during text extraction: {e}")
        return None

# Function to create FAISS vectors from multiple CSV files
def create_faiss_from_multiple_csv(csv_files):
    try:
        documents = []
        for csv_file in csv_files:
            # Load text from each CSV
            df = pd.read_csv(csv_file)
            st.write(f"Loaded {len(df)} rows from {csv_file}.")

            if df.empty:
                st.warning(f"{csv_file} is empty. Skipping FAISS vector creation.")
                continue

            # Handle missing or invalid values in the Content column
            df["Content"] = df["Content"].fillna("No content available")  # Replace NaN with placeholder text
            df = df[df["Content"].str.strip() != ""]  # Drop rows with empty or whitespace-only content

            # Convert rows to Document objects for chunking
            documents.extend([ 
                Document(page_content=row["Content"], metadata={"document_name": row["Document Name"]})
                for _, row in df.iterrows()
            ]) 

        if not documents:
            st.warning("No documents found after processing CSVs. FAISS vector store will not be created.")
            return

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        final_documents = text_splitter.split_documents(documents)
        st.write(f"Processed {len(final_documents)} document chunks.")

        # Create FAISS vector store
        embeddings = NVIDIAEmbeddings()
        vectors = FAISS.from_documents(final_documents, embeddings)
        
        # Save FAISS index to disk manually using faiss.write_index
        faiss_file = os.path.join(SAVED_DATA_DIR, "faiss_index.index")
        faiss.write_index(vectors.index, faiss_file)  # Save the FAISS index
        st.session_state.vectors = vectors  # Store in session state
        st.success(f"FAISS vector store created successfully and saved to {faiss_file}!")

    except Exception as e:
        st.error(f"Error during FAISS vector creation: {e}")

# Function to load previously saved FAISS vector store and CSV files
def load_saved_data():
    faiss_file = os.path.join(SAVED_DATA_DIR, "faiss_index.index")
    
    if os.path.exists(faiss_file):
        # Load the FAISS index manually using faiss.read_index
        index = faiss.read_index(faiss_file)
        
        # Create the embeddings and docstore objects (these should be the same as when you created the index)
        embedding_function = NVIDIAEmbeddings()  # Make sure this is the same function used earlier
        docstore = {}  # Dummy docstore - you would need to implement a way to retrieve docs
        index_to_docstore_id = {}  # Dummy mapping, you will need to map ids to docstore if saved earlier
        
        # Create FAISS vector store with the loaded index and other metadata
        vectors = FAISS(embedding_function, docstore, index_to_docstore_id, index)

        # Store the vectors in session state
        st.session_state.vectors = vectors
        st.success("Loaded saved FAISS vector store.")

    else:
        st.warning("No saved FAISS vector store found.")

# Text input for user query
prompt1 = st.text_input("Enter your question based on the uploaded documents:")

# Upload new PDFs button
uploaded_pdfs = st.file_uploader("Upload PDFs to process", type="pdf", accept_multiple_files=True)

# Button to extract text and create embeddings
if st.button("Extract Text and Create Embeddings") or uploaded_pdfs:
    if uploaded_pdfs:
        # Save uploaded PDFs to a temporary directory
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)

        pdf_files = []
        for uploaded_file in uploaded_pdfs:
            pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_files.append(pdf_path)

        # Step 1: Extract text from PDFs and create separate CSVs
        csv_file = extract_text_to_csv(pdf_files)
        if csv_file:
            # Step 2: Create FAISS vectors from the CSV files
            create_faiss_from_multiple_csv([csv_file])

        # Clean up the temporary PDFs directory
        shutil.rmtree(temp_dir)

# Build the document retrieval chain
prompt_template = ChatPromptTemplate.from_template(""" 
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""")
# Load saved FAISS vector store on app start
if "vectors" not in st.session_state:
    load_saved_data()

# Process user query
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please generate document embeddings first by clicking 'Extract Text and Create Embeddings'.")
    else:
        try:
            # Ensure retriever is correctly created
            retriever = st.session_state.vectors.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # Use similarity search
            
            
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Query the chain and measure response time
            start_time = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            end_time = time.process_time()
            st.write(f"Response Time: {end_time - start_time:.2f} seconds")

            # Display the LLM response
            st.subheader("Response:")
            st.write(response.get('answer', 'No response found.'))

            # Display similar documents in an expander
            with st.expander("Document Similarity Search"):
                if "context" in response:
                    for i, doc in enumerate(response["context"]):
                        st.write(f"**Document {i + 1}:**")
                        st.write(doc.page_content)
                        st.write("--------------------------------")
                else:
                    st.write("No similar documents found.")

        except Exception as e:
            st.error(f"Error during query processing: {e}")
