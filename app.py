import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. UI Setup
st.set_page_config(page_title="Gita AI", page_icon="🙏")
st.title("🙏 Bhagavad Gita AI Expert")

# 2. Memory & Brain Initialization
# We use a free, lightweight embedding model from HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Pinecone.from_existing_index("gita-index", embeddings)

# Use Groq for lightning-fast, free inference
llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.2)

# 3. The Scholar Prompt (Internal Validation Agent)
template = """You are a senior Vedic scholar. 
Answer the user's question ONLY based on the provided Bhagavad Gita excerpts.

CONTEXT: {context}
QUESTION: {question}

INSTRUCTIONS FOR YOUR RESPONSE:
1. Start with a relevant Sanskrit/English quote from the context.
2. Provide a clear, compassionate explanation.
3. If the answer isn't in the context, say: "I apologize, but this specific wisdom is not in my current Gita database."

ANSWER:"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# 4. Retrieval-Augmented Generation (RAG) Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# 5. User Interaction
query = st.text_input("What is your question for the Gita?")
if query:
    with st.spinner("Consulting the scriptures..."):
        result = qa_chain({"query": query})
        
        st.markdown(result["result"])
        
        # Validation: Explicitly showing the quote used to ground the answer
        with st.expander("View Source Verses (Validation)"):
            for doc in result["source_documents"]:
                st.write(f"📖 {doc.page_content}")
