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

# Helper function to ensure directory exists
def ensure_directory(directory):
    os.makedirs(directory, exist_ok=True)

# Function to extract text from PDFs and save as CSV
def extract_text_to_csv(pdf_directory, csv_folder):
    try:
        if not os.path.isdir(pdf_directory):
            st.error(f"Directory {pdf_directory} does not exist.")
            return None

        loader = PyPDFDirectoryLoader(pdf_directory)
        documents = loader.load()
        if not documents:
            st.warning("No PDF files found in the directory.")
            return None

        # Extract content and metadata
        data = [{"Document Name": doc.metadata.get("source", f"Document_{i + 1}"), "Content": doc.page_content}
                for i, doc in enumerate(documents)]
        
        # Save to a unique CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(csv_folder, f"extracted_text_{timestamp}_{os.path.basename(pdf_directory)}.csv")
        pd.DataFrame(data).to_csv(csv_file, index=False)
        st.success(f"Extracted text saved to {csv_file}")
        return csv_file

    except Exception as e:
        st.error(f"Error during text extraction: {e}")
        return None

# Function to combine multiple CSV files
def combine_csv_files(csv_folder, combined_csv_file):
    try:
        csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith(".csv")]
        if not csv_files:
            st.warning("No CSV files found to combine.")
            return None

        combined_data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        combined_data.to_csv(combined_csv_file, index=False)
        st.success(f"Combined data saved to {combined_csv_file}")
        return combined_csv_file

    except Exception as e:
        st.error(f"Error while combining CSV files: {e}")
        return None

# Function to create FAISS vectors from a combined CSV file
def create_faiss_from_combined_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
        st.write(f"Loaded {len(df)} rows from {csv_file}.")
        
        df["Content"] = df["Content"].fillna("No content available")
        df = df[df["Content"].str.strip() != ""]
        documents = [
            Document(page_content=row["Content"], metadata={"document_name": row["Document Name"]})
            for _, row in df.iterrows()
        ]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        final_documents = text_splitter.split_documents(documents)
        st.write(f"Processed {len(final_documents)} document chunks.")

        embeddings = NVIDIAEmbeddings()
        vectors = FAISS.from_documents(final_documents, embeddings)
        st.session_state.vectors = vectors
        st.success("FAISS vector store created successfully!")

    except Exception as e:
        st.error(f"Error during FAISS vector creation: {e}")

# Button to extract text and create embeddings
csv_folder = "./csv_files"
ensure_directory(csv_folder)
pdf_directory = "./dgx"
combined_csv_file = os.path.join(csv_folder, "combined_data.csv")

if st.button("Extract Text and Create Embeddings"):
    try:
        # Step 1: Extract text from PDFs and save as CSV
        extract_text_to_csv(pdf_directory, csv_folder)

        # Step 2: Combine all CSV files in the folder
        if os.path.isdir(csv_folder):
            combined_csv_path = combine_csv_files(csv_folder, combined_csv_file)
            
            # Step 3: Create FAISS vectors from the combined CSV
            if combined_csv_path:
                create_faiss_from_combined_csv(combined_csv_path)

    except Exception as e:
        st.error(f"Error during processing: {e}")

# Text input for user query
prompt1 = st.text_input("Enter your question based on the uploaded documents:")

# Process user query
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please generate document embeddings first by clicking 'Extract Text and Create Embeddings'.")
    else:
        try:
            # Build retrieval and question-answer chain
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

            start_time = time.process_time()
            response = retrieval_chain.invoke({'input': prompt1})
            end_time = time.process_time()
            st.write(f"Response Time: {end_time - start_time:.2f} seconds")

            # Display the LLM response
            st.subheader("Response:")
            st.write(response.get('answer', 'No response found.'))

            # Display similar documents
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
