import logging
import sys
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure logging to output to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Attempting to initialize HuggingFaceEmbedding...")
        # Initialize the embedding model
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.embed_model = embed_model
        logger.info("HuggingFaceEmbedding initialized successfully.")
        
        test_text = "This is a test sentence."
        logger.info(f"Attempting to embed text: '{test_text}'")
        
        # Get the embedding for the test text
        embedding = embed_model.get_text_embedding(test_text)
        
        logger.info(f"Successfully generated embedding of dimension: {len(embedding)}")
        logger.info("Local embedding model seems to be working correctly.")

    except Exception as e:
        logger.error("An error occurred during the embedding test:", exc_info=True)

if __name__ == "__main__":
    main()