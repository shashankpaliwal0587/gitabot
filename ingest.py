import os
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# 1. Setup API Key
# Run 'export PINECONE_API_KEY="your_key"' in your terminal first
pc_api_key = os.getenv("PINECONE_API_KEY")
if not pc_api_key:
    raise ValueError("Please set PINECONE_API_KEY in your environment variables.")

# 2. Load and Process CSV
df = pd.read_csv('bhagavad_gita_verses.csv')

documents = []
for index, row in df.iterrows():
    # Constructing a rich 'page_content' so the LLM understands the context
    # Example: "Gita 1.1 (Arjun Viṣhād Yog): Dhritarashtra said..."
    text_content = f"Gita {row['chapter_verse']} ({row['chapter_title']}): {row['translation']}"
    
    # We store the metadata so the Chatbot can filter or cite precisely
    metadata = {
        "chapter": row['chapter_number'],
        "verse": row['chapter_verse'],
        "title": row['chapter_title']
    }
    
    doc = Document(page_content=text_content, metadata=metadata)
    documents.append(doc)

# 3. Create Embeddings
# We use the same model as in app.py to ensure the 'math' matches
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Upload to Pinecone
print(f"Uploading {len(documents)} verses to Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents, 
    embeddings, 
    index_name="gita-index"
)

print("✅ Success! Your Gita knowledge base is now indexed with metadata.")