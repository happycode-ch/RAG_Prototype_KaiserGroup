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
            # Preprocess query to identify key terms
            key_terms = {
                'cto': ['cto', 'chief technology officer', 'technology officer', 'tech officer'],
                'location': ['headquarters', 'office', 'located', 'location', 'where'],
                'project': ['project', 'budget', 'alpha', 'cost', 'million'],
                'employee': ['who', 'employee', 'staff', 'team', 'lead'],
            }
            
            query_lower = query.lower()
            relevant_terms = []
            query_categories = set()
            
            # Find matching terms and categories
            for category, terms in key_terms.items():
                if any(term in query_lower for term in terms):
                    relevant_terms.extend(terms)
                    query_categories.add(category)
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            
            # Search vector store with no threshold filtering
            results = self.vector_store.search(
                "documents",
                query_embedding,
                threshold=None  # Disable threshold filtering
            )
            
            if results and results['documents']:
                # Enhance results with semantic matching
                enhanced_results = []
                for doc, score, metadata in zip(results['documents'], results['distances'], results['metadatas']):
                    doc_lower = doc.lower()
                    
                    # Calculate term boost (0.2 per matching term)
                    term_matches = sum(1 for term in relevant_terms if term in doc_lower)
                    term_boost = min(0.6, term_matches * 0.2)  # Cap at 0.6
                    
                    # Calculate section boost
                    section_boost = 0.0
                    if metadata and 'section' in metadata:
                        section = metadata['section'].lower()
                        
                        # Direct section match
                        if any(category in section for category in query_categories):
                            section_boost += 0.3
                        
                        # Content relevance boost
                        if any(term in section for term in relevant_terms):
                            section_boost += 0.2
                    
                    # Context boost for specific types of queries
                    context_boost = 0.0
                    if 'who' in query_lower and 'employee information' in doc_lower:
                        context_boost = 0.2
                    elif 'where' in query_lower and 'office locations' in doc_lower:
                        context_boost = 0.2
                    elif 'budget' in query_lower and 'project' in doc_lower:
                        context_boost = 0.2
                    
                    # Calculate base similarity (convert distance to similarity)
                    base_similarity = max(0, 1 - score)
                    
                    # Calculate final score with weighted components
                    final_score = min(1.0, base_similarity + term_boost + section_boost + context_boost)
                    
                    enhanced_results.append({
                        'document': doc,
                        'score': final_score,
                        'metadata': metadata,
                        'term_boost': term_boost,
                        'section_boost': section_boost,
                        'context_boost': context_boost,
                        'base_similarity': base_similarity
                    })
                
                # Sort by final score (highest first)
                enhanced_results.sort(key=lambda x: x['score'], reverse=True)
                
                # Update the results
                results = {
                    'documents': [r['document'] for r in enhanced_results],
                    'distances': [1 - r['score'] for r in enhanced_results],  # Convert back to distances
                    'metadatas': [r['metadata'] for r in enhanced_results],
                    'debug_info': [{
                        'base_similarity': r['base_similarity'],
                        'term_boost': r['term_boost'],
                        'section_boost': r['section_boost'],
                        'context_boost': r['context_boost'],
                        'final_score': r['score']
                    } for r in enhanced_results]
                }
            
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
                
                for doc, score, debug in zip(results['documents'], results['distances'], results['debug_info']):
                    st.markdown("---")
                    similarity_score = debug['final_score']  # Use the enhanced score directly
                    
                    # Color code the relevance score
                    if similarity_score >= 0.8:
                        score_color = "green"
                    elif similarity_score >= 0.5:
                        score_color = "orange"
                    else:
                        score_color = "red"
                    
                    # Extract relevant snippet
                    sentences = doc.split('. ')
                    relevant_snippet = ""
                    query_terms = query.lower().split()
                    
                    for sentence in sentences:
                        sentence_lower = sentence.lower()
                        # Skip section headers and metadata
                        if ':' in sentence and any(header in sentence_lower for header in ['overview', 'information:', 'locations:', 'projects:']):
                            continue
                        if any(term in sentence_lower for term in query_terms):
                            # Clean up the sentence
                            relevant_snippet = sentence.split(': ')[-1].strip() + "."
                            break
                    
                    if not relevant_snippet:
                        # If no direct match, try to find a sentence with semantic relevance
                        for sentence in sentences:
                            sentence_lower = sentence.lower()
                            if any(category in sentence_lower for category in query_categories):
                                relevant_snippet = sentence.split(': ')[-1].strip() + "."
                                break
                    
                    if not relevant_snippet:
                        relevant_snippet = sentences[0].split(': ')[-1].strip() + "." if sentences else doc
                    
                    st.markdown(f"**Relevance Score:** :{score_color}[{similarity_score:.2f}]")
                    st.markdown(f"**Most Relevant Part:**\n{relevant_snippet}")
                    
                    # Show full content in expander
                    with st.expander("Show Full Content"):
                        st.markdown(doc)
                    
                    # Add debug info in an expander
                    with st.expander("Debug Info"):
                        st.text(f"Base similarity: {debug['base_similarity']:.2f}")
                        st.text(f"Term boost: {debug['term_boost']:.2f}")
                        st.text(f"Section boost: {debug['section_boost']:.2f}")
                        st.text(f"Context boost: {debug['context_boost']:.2f}")
                        st.text(f"Final score: {debug['final_score']:.2f}")
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