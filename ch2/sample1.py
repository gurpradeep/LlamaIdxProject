from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import traceback
from dotenv import load_dotenv

load_dotenv()

try:

    current_directory = os.path.dirname(os.path.abspath(__file__))
    files = os.path.join(current_directory, 'files')
    documents = SimpleDirectoryReader(files).load_data()
    hf_embedding = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index = VectorStoreIndex.from_documents(documents,embed_model=hf_embedding, llm=None)
    query_engine = index.as_query_engine()
    response = query_engine.query("summarize each document in a few sentences")

    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
    traceback.print_exc()
