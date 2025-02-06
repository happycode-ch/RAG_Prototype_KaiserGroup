import streamlit as st
import os
import sys
from pathlib import Path
import configparser
from loguru import logger

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.document_processor import DocumentProcessor
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore

# Configure logger
logger.add("logs/app.log", rotation="1 day")

class RAGApp:
    def __init__(self):
        """Initialize the RAG application."""
        self.config = configparser.ConfigParser()
        self.config.read("config/config.ini")
        
        self.processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        
        # Initialize session state
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = set()
        
        # Create data directories
        os.makedirs("data/raw", exist_ok=True)
    
    def process_uploaded_file(self, uploaded_file):
        """Process an uploaded PDF file."""
        if uploaded_file.name in st.session_state.processed_files:
            st.warning(f"File {uploaded_file.name} has already been processed.")
            return
        
        # Save the uploaded file
        temp_path = f"data/raw/{uploaded_file.name}"
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Process the document
            chunks = self.processor.process_document(temp_path)
            
            # Generate embeddings
            result = self.embedding_generator.generate_embeddings_with_metadata(chunks)
            
            # Store in vector database
            collection_name = "documents"
            
            # Delete existing collection if it exists
            try:
                self.vector_store.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception as e:
                logger.info(f"No existing collection to delete: {str(e)}")
            
            # Add documents to fresh collection
            self.vector_store.add_documents(
                collection_name,
                result['texts'],
                result['embeddings'],
                result['metadata']
            )
            
            st.session_state.processed_files.add(uploaded_file.name)
            st.success(f"Successfully processed {uploaded_file.name}")
            st.info(f"Original file saved at: {temp_path}")
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
            # Clean up file only on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def search_documents(self, query: str):
        """Search for relevant documents based on the query."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            
            # Search vector store with no threshold filtering
            results = self.vector_store.search(
                "documents",
                query_embedding,
                threshold=None  # Disable threshold filtering
            )
            
            return results
            
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
            logger.error(f"Search error: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="RAG Prototype",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("📚 RAG Prototype")
    st.markdown("""
    This is a simple RAG (Retrieval-Augmented Generation) prototype that allows you to:
    1. Upload PDF documents
    2. Process and index their contents
    3. Search through the documents using natural language queries
    """)
    
    app = RAGApp()
    
    # File upload section
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            app.process_uploaded_file(uploaded_file)
    
    # Search section
    st.header("🔍 Search Documents")
    query = st.text_input("Enter your search query")
    
    if query:
        with st.spinner("Searching..."):
            results = app.search_documents(query)
            
            if results and results['documents']:
                st.subheader("Search Results")
                
                for doc, score in zip(results['documents'], results['distances']):
                    st.markdown("---")
                    # Ensure score is between 0 and 1
                    similarity_score = max(0, min(1, 1 - (score / 2)))
                    
                    # Color code the relevance score
                    if similarity_score >= 0.7:
                        score_color = "green"
                    elif similarity_score >= 0.4:
                        score_color = "orange"
                    else:
                        score_color = "red"
                    
                    # Extract relevant snippet
                    sentences = doc.split('. ')
                    relevant_snippet = ""
                    for sentence in sentences:
                        if "CTO" in sentence or "Chief Technology Officer" in sentence:
                            relevant_snippet = sentence + "."
                            break
                    if not relevant_snippet:
                        relevant_snippet = doc
                    
                    st.markdown(f"**Relevance Score:** :{score_color}[{similarity_score:.2f}]")
                    st.markdown(f"**Most Relevant Part:**\n{relevant_snippet}")
                    
                    # Show full content in expander
                    with st.expander("Show Full Content"):
                        st.markdown(doc)
                    
                    # Add debug info in an expander
                    with st.expander("Debug Info"):
                        st.text(f"Raw distance score: {score}")
                        st.text(f"Adjusted similarity score: {similarity_score}")
            else:
                st.info("No results found in the vector store. Try rephrasing your query.")
    
    # Status section
    st.sidebar.header("📊 Status")
    st.sidebar.metric(
        "Processed Documents",
        len(st.session_state.processed_files)
    )
    
    if st.session_state.processed_files:
        st.sidebar.markdown("**Processed Files:**")
        for file in st.session_state.processed_files:
            st.sidebar.markdown(f"- {file}")

if __name__ == "__main__":
    main() 