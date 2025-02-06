from typing import List, Dict, Any, Optional
import chromadb
import numpy as np
import configparser
from loguru import logger
import os

class VectorStore:
    def __init__(self, config_path: str = "config/config.ini"):
        """Initialize the vector store with configuration."""
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        self.db_path = self.config.get('DEFAULT', 'vector_db_path')
        self.similarity_threshold = self.config.getfloat('RETRIEVAL', 'similarity_threshold')
        self.top_k = self.config.getint('RETRIEVAL', 'top_k')
        
        # Ensure the database directory exists
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize ChromaDB with new configuration
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        logger.info(f"Initialized vector store at: {self.db_path}")
    
    def create_collection(self, name: str) -> Any:
        """Create a new collection or get existing one."""
        try:
            collection = self.client.create_collection(name)
        except ValueError:
            # Collection already exists
            collection = self.client.get_collection(name)
            logger.info(f"Using existing collection: {name}")
        return collection
    
    def add_documents(
        self,
        collection_name: str,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents and their embeddings to the vector store."""
        collection = self.create_collection(collection_name)
        
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        
        if metadata is None:
            metadata = [{} for _ in texts]
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            collection.add(
                documents=texts[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadata[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        logger.info(f"Added {len(texts)} documents to collection: {collection_name}")
    
    def search(
        self,
        collection_name: str,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, List]:
        """Search for similar documents using the query embedding."""
        if top_k is None:
            top_k = self.top_k
        
        collection = self.client.get_collection(collection_name)
        
        # Get raw results without filtering
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        # Only filter if threshold is explicitly provided
        if threshold is not None:
            distances = results['distances'][0]
            mask = [d <= threshold for d in distances]
            
            filtered_results = {
                'ids': [results['ids'][0][i] for i, m in enumerate(mask) if m],
                'documents': [results['documents'][0][i] for i, m in enumerate(mask) if m],
                'metadatas': [results['metadatas'][0][i] for i, m in enumerate(mask) if m],
                'distances': [d for i, d in enumerate(distances) if mask[i]]
            }
            
            # Log search results for debugging
            logger.info(f"Search query returned {len(filtered_results['documents'])} results after filtering")
            for i, (doc, dist) in enumerate(zip(filtered_results['documents'], filtered_results['distances'])):
                logger.info(f"Result {i+1} - Score: {1-dist:.2f}")
                logger.info(f"Content: {doc[:100]}...")
            
            return filtered_results
        
        # Return unfiltered results with logging
        logger.info(f"Search query returned {len(results['documents'][0])} results")
        for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
            logger.info(f"Result {i+1} - Score: {1-dist:.2f}")
            logger.info(f"Content: {doc[:100]}...")
        
        return {
            'ids': results['ids'][0],
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection from the vector store."""
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
        except ValueError as e:
            logger.warning(f"Collection {name} not found: {str(e)}")

if __name__ == "__main__":
    # Example usage
    store = VectorStore()
    
    # Test with sample data
    collection_name = "test_collection"
    texts = ["Sample document 1", "Sample document 2"]
    embeddings = np.random.rand(2, 384)  # Example embedding dimension
    
    # Add documents
    store.add_documents(collection_name, texts, embeddings)
    
    # Test search
    query_embedding = np.random.rand(384)
    results = store.search(collection_name, query_embedding)
    logger.info(f"Found {len(results['documents'])} matching documents") 