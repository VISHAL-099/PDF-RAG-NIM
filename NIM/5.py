import streamlit as st
import os
import pandas as pd
import time
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

# Path to the combined CSV file
combined_csv_file = "./csv_files/combined_data.csv"

# Function to create FAISS vectors from a combined CSV file
def create_faiss_from_combined_csv(csv_file):
    try:
        if not os.path.exists(csv_file):
            st.error(f"Combined CSV file '{csv_file}' not found.")
            return False

        # Load and process the combined CSV file
        df = pd.read_csv(csv_file)
        st.write(f"Loaded {len(df)} rows from {csv_file}.")

        # Ensure content column is clean and non-empty
        df["Content"] = df["Content"].fillna("No content available")
        df = df[df["Content"].str.strip() != ""]
        documents = [
            Document(page_content=row["Content"], metadata={"document_name": row["Document Name"]})
            for _, row in df.iterrows()
        ]

        # Split text into chunks for better embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        final_documents = text_splitter.split_documents(documents)
        st.write(f"Processed {len(final_documents)} document chunks.")

        # Create FAISS vector store
        embeddings = NVIDIAEmbeddings()
        vectors = FAISS.from_documents(final_documents, embeddings)
        st.session_state.vectors = vectors  # Store in session state
        st.success("FAISS vector store created successfully!")
        return True

    except Exception as e:
        st.error(f"Error during FAISS vector creation: {e}")
        return False

# Initialize FAISS vectors when the server starts
if "vectors" not in st.session_state:
    st.write("Initializing FAISS vector store...")
    if create_faiss_from_combined_csv(combined_csv_file):
        st.success("FAISS vectors initialized successfully!")
    else:
        st.error("Failed to initialize FAISS vectors. Please ensure the combined CSV file exists.")

# Text input for user query
prompt1 = st.text_input("Enter your question based on the uploaded documents:")

if st.button("Extract Text and Create Embeddings"):
    # Step 1: Extract text from PDFs to CSV
    extract_text_to_csv(pdf_directory, csv_file)

    # Step 2: Create FAISS vectors from the CSV
    create_faiss_from_csv(csv_file)
# Process user query
if prompt1:
    if "vectors" not in st.session_state:
        st.warning("FAISS vector store is not initialized. Please restart the server with a valid combined CSV file.")
    else:
        try:
            # Build the retrieval and question-answer chain
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
