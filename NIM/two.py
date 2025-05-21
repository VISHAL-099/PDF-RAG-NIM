import streamlit as st
import os
import pandas as pd
import time
from datetime import datetime
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
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

# Function to extract text from PDFs and save to multiple CSV files
def extract_text_to_csv(pdf_directory):
    data = []
    try:
        # Check if the directory exists and if there are any files inside
        if not os.path.isdir(pdf_directory):
            st.error(f"Directory {pdf_directory} does not exist.")
            return None

        # Load PDF files
        loader = PyPDFDirectoryLoader(pdf_directory)
        documents = loader.load()
        if len(documents) == 0:
            st.warning("No PDF files found in the directory.")
            return None
        
        st.write(f"Loaded {len(documents)} documents.")

        # Extract content and metadata from each document
        for i, doc in enumerate(documents):
            data.append({"Document Name": doc.metadata.get("source", f"Document_{i + 1}"), "Content": doc.page_content})

        # Generate a unique file name with a timestamp for each directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"extracted_text_{timestamp}_{os.path.basename(pdf_directory)}.csv"

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

            # Handle missing or invalid values in the Content column
            df["Content"] = df["Content"].fillna("No content available")  # Replace NaN with placeholder text
            df = df[df["Content"].str.strip() != ""]  # Drop rows with empty or whitespace-only content

            # Convert rows to Document objects for chunking
            documents.extend([ 
                Document(page_content=row["Content"], metadata={"document_name": row["Document Name"]})
                for _, row in df.iterrows()
            ])

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        final_documents = text_splitter.split_documents(documents)
        st.write(f"Processed {len(final_documents)} document chunks.")

        # Create FAISS vector store
        embeddings = NVIDIAEmbeddings()
        vectors = FAISS.from_documents(final_documents, embeddings)
        st.session_state.vectors = vectors  # Store in session state
        st.success("FAISS vector store created successfully!")

    except Exception as e:
        st.error(f"Error during FAISS vector creation: {e}")

# Text input for user query
prompt1 = st.text_input("Enter your question based on the uploaded documents:")

# Button to extract text and create embeddings
if st.button("Extract Text and Create Embeddings"):
    pdf_directory = "./dgx"  # Path to the directory containing PDFs
    all_csv_files = []  # List to store all generated CSV files

    # Step 1: Extract text from PDFs and create separate CSVs
    csv_file = extract_text_to_csv(pdf_directory)
    if csv_file:
        all_csv_files.append(csv_file)  # Add the CSV file to the list

    # Step 2: Create FAISS vectors from the CSV files
    if all_csv_files:
        create_faiss_from_multiple_csv(all_csv_files)

# Process user query
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please generate document embeddings first by clicking 'Extract Text and Create Embeddings'.")
    else:
        try:
            # Build the document retrieval chain
            prompt_template = ChatPromptTemplate.from_template(""" 
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
""")
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retriever = st.session_state.vectors.as_retriever()
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
