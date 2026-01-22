import os
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 1. Verification Logic
# Ensure the index name matches exactly what you used in ingest.py
INDEX_NAME = "gita-index"

# 2. Setup Embeddings
# CRITICAL: This must be the EXACT same model used in ingest.py
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Connect to the CORRECT existing index
# We use from_existing_index to ensure we aren't creating a blank slate
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME, 
    embedding=embeddings
)

# 4. Setup the LLM (Groq)
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.3-70b-versatile", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 5. The Expert Prompt (With Validation)
template = """You are a Bhagavad Gita scholar. Answer the question using ONLY the context.
If you don't know, say you don't know.

CONTEXT: {context}
QUESTION: {question}

YOUR RESPONSE MUST:
1. Start with the direct verse/quote.
2. Provide the English explanation.
3. Cite the Chapter and Verse number found in the context.

ANSWER:"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# 6. Build the Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # Fetches top 3 verses
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# --- Streamlit UI ---
st.title("🙏 Bhagavad Gita Expert")
user_query = st.text_input("Ask a question about life or duty:")

if user_query:
    with st.spinner("Searching the verses..."):
        response = qa_chain({"query": user_query})
        
        # Display the AI's Answer
        st.markdown(response["result"])
        
        # Validation: Show the raw data that backed up the answer
        with st.expander("Validation: Verses used for this answer"):
            for doc in response["source_documents"]:
                # Accessing the metadata we stored in ingest.py
                st.write(f"**{doc.metadata['chapter']} - Verse {doc.metadata['verse']}**")
                st.info(doc.page_content)