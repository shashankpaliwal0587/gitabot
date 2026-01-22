import os
import time
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 1. Setup API Key and Client
pc_api_key = os.getenv("PINECONE_API_KEY")
if not pc_api_key:
    raise ValueError("❌ Please run 'export PINECONE_API_KEY=your_key' in your terminal.")

pc = Pinecone(api_key=pc_api_key)
index_name = "gita-index"

# 2. Automatically Create Index if it doesn't exist
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"🚀 Creating new index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384, # Must match all-MiniLM-L6-v2 dimensions
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1" # Free tier usually lives in us-east-1
        )
    )
    # Wait for index to be initialized
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print("✅ Index is ready!")
else:
    print(f"ℹ️ Index '{index_name}' already exists. Skipping creation.")

# 3. Load and Process CSV
print("📖 Reading Gita CSV...")
df = pd.read_csv('bhagavad_gita_verses.csv')

documents = []
for index, row in df.iterrows():
    # Construct content for the LLM to read
    text_content = f"Gita {row['chapter_verse']} ({row['chapter_title']}): {row['translation']}"
    
    # Store clean metadata for citations
    metadata = {
        "chapter": str(row['chapter_number']),
        "verse": str(row['chapter_verse']),
        "title": str(row['chapter_title'])
    }
    
    doc = Document(page_content=text_content, metadata=metadata)
    documents.append(doc)

# 4. Create Embeddings & Upload
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print(f"📤 Uploading {len(documents)} verses to Pinecone. Please wait...")
vectorstore = PineconeVectorStore.from_documents(
    documents, 
    embeddings, 
    index_name=index_name
)

print("🎉 Success! Your knowledge base is fully indexed and ready for the chatbot.")