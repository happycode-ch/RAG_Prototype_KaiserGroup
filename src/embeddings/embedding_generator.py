from typing import List, Dict, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import configparser
from loguru import logger

class EmbeddingGenerator:
    def __init__(self, config_path: str = "config/config.ini"):
        """Initialize the embedding generator with configuration."""
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        model_name = self.config.get('DEFAULT', 'embedding_model')
        self.batch_size = self.config.getint('EMBEDDINGS', 'batch_size')
        self.show_progress = self.config.getboolean('EMBEDDINGS', 'show_progress')
        self.normalize = self.config.getboolean('EMBEDDINGS', 'normalize_embeddings')
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            raise ValueError("No texts provided for embedding generation")
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size),
                     disable=not self.show_progress):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True  # Always normalize
            )
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        
        # Double-check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            logger.warning("Re-normalizing embeddings")
            embeddings = embeddings / norms[:, np.newaxis]
        
        return embeddings
    
    def generate_embeddings_with_metadata(
        self,
        chunks: List[Dict[str, str]]
    ) -> Dict[str, Union[List[str], np.ndarray]]:
        """Generate embeddings for chunks with their metadata."""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        return {
            'texts': texts,
            'embeddings': embeddings,
            'metadata': chunks
        }
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query text."""
        embedding = self.model.encode(
            query,
            normalize_embeddings=True  # Always normalize
        )
        
        # Double-check normalization
        norm = np.linalg.norm(embedding)
        if not np.isclose(norm, 1.0, atol=1e-5):
            logger.warning("Re-normalizing query embedding")
            embedding = embedding / norm
        
        return embedding

if __name__ == "__main__":
    # Example usage
    generator = EmbeddingGenerator()
    
    # Test with some sample texts
    sample_texts = [
        "This is a sample document for testing embeddings.",
        "Another example text to process.",
        "Testing the embedding generator functionality."
    ]
    
    embeddings = generator.generate_embeddings(sample_texts)
    logger.info(f"Generated embeddings shape: {embeddings.shape}") 