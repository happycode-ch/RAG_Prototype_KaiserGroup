# RAG Prototype

A Retrieval-Augmented Generation (RAG) system built with open-source tools. This prototype demonstrates semantic search capabilities using local embeddings and vector storage.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-prototype
   ```

2. Create and activate a virtual environment:
   ```bash
   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create necessary directories:
   ```bash
   mkdir -p data/raw logs chroma_db
   ```

## Running the Application

1. Start the Streamlit web interface:
   ```bash
   streamlit run src/web/app.py
   ```
   The application will be available at http://localhost:8501

## Testing the System

1. Generate a test document:
   ```bash
   python src/data/create_test_doc.py
   ```
   This creates a sample PDF with known content at `data/raw/test_data.pdf`

2. Using the Web Interface:
   - Open http://localhost:8501 in your browser
   - Click "Browse files" in the Upload Documents section
   - Select the generated `test_data.pdf`
   - Wait for the "Successfully processed" message

3. Try these test queries:
   - "Who is the CTO?"
   - "Where is the company headquarters?"
   - "What is Project Alpha's budget?"

## Understanding the Results

The search results will show:
- A relevance score (color-coded):
  - Green (≥0.8): Highly relevant
  - Orange (≥0.5): Moderately relevant
  - Red (<0.4): Less relevant
- The most relevant snippet, automatically extracted and cleaned from the document
- Full content (expandable)
- Debug information showing how the score was calculated:
  - Base similarity: Core semantic matching score
  - Term boost: Additional score for matching specific terms
  - Section boost: Score for matching relevant document sections
  - Context boost: Score for matching question types with content

The system uses multiple strategies to find the most relevant information:
1. Semantic search using document embeddings
2. Term-based matching for specific keywords
3. Section-aware context understanding
4. Question-type analysis (e.g., who, where, what)

## System Components

- `src/web/app.py`: Streamlit web interface
- `src/data/document_processor.py`: PDF processing and text chunking
- `src/embeddings/embedding_generator.py`: Text embedding generation
- `src/retrieval/vector_store.py`: Vector storage and similarity search

## Configuration

Key settings in `config/config.ini`:
- `chunk_size`: Text chunk size (default: 200)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `embedding_model`: Model for generating embeddings (default: sentence-transformers/all-MiniLM-L6-v2)
- `similarity_threshold`: Minimum similarity score (default: 0.5)

## Troubleshooting

1. If you get "No relevant documents found":
   - Check if the document was processed successfully
   - Try rephrasing your query
   - Check the debug information for similarity scores

2. If you get embedding-related errors:
   - Ensure you have enough RAM (at least 4GB recommended)
   - Check your internet connection (first run downloads the embedding model)

3. If the application crashes:
   - Check the logs in the `logs` directory
   - Ensure all dependencies are installed correctly
   - Verify Python version compatibility

## Limitations

- Currently supports PDF files only
- Processes one document at a time
- Keeps the entire embedding model in memory
- Simple text chunking strategy
- Basic similarity search without reranking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [PDFPlumber](https://github.com/jsvine/pdfplumber) 