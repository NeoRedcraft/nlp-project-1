import streamlit as st
import os
import tempfile

from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

# Page Configuration
st.set_page_config(page_title="RAG Mental Health Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Retrieval-Augmented Generation (RAG) Chatbot")
st.markdown("Ask anything based on the provided mental health documents.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Pinecone Configuration
try:
    pinecone_api_key = st.secrets.get("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
    index_name = st.secrets.get("PINECONE_INDEX_NAME", os.getenv("PINECONE_INDEX_NAME", "mental-health-rag"))
except Exception:
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "mental-health-rag")

if not pinecone_api_key:
    st.error("Pinecone API key not found! Please set PINECONE_API_KEY in your secrets or `.env`.")
    st.stop()

@st.cache_resource(show_spinner="Loading Embedding Model (Sentence-Transformers)...")
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Downloading and Loading Qwen Language Model (This may take a few minutes)...")
def load_llm():
    # Load Qwen Locally natively to the space avoiding Serverless Endpoints
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    try:
        # device=-1 forces CPU execution. Essential for HF Free Tier Spaces.
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            device=-1,
            pipeline_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.1,
                "top_k": 30,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            }
        )
        return llm
    except Exception as e:
        import traceback
        st.error(f"Failed to load local Hugging Face pipeline:\n\n{traceback.format_exc()}")
        st.stop()

# Load Models
embeddings = load_embedding_model()
try:
    llm = load_llm()
except Exception as e:
    st.error(f"Error loading LLM: {e}")
    st.stop()

# Sidebar Information
with st.sidebar:
    st.header("📄 Knowledge Base")
    st.success("✅ Mental Health guiding documents are currently loaded and active in the Pinecone Vector Store.")
    st.info("You can begin asking questions about mental health right away!")

# Helper function to get retriever
def get_retriever():
    try:
        vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key)
        return vectorstore.as_retriever(search_kwargs={"k": 6})
    except Exception as e:
        st.sidebar.error(f"Failed to connect to Pinecone: {e}")
        return None

retriever = get_retriever()

# Initialize QA Chain
qa_chain = None
if retriever:
    prompt_template = """<|im_start|>system
You are a document question-answering system. /no_think

Rules:
- Do NOT provide options.
- Do NOT explain reasoning.
- Do NOT analyze the context.
- Do NOT restate the context.
- Provide a direct answer only.
<|im_end|>
<|im_start|>user
Context:
{context}

Question:
{question}
<|im_end|>
<|im_start|>assistant
Final Answer:
"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    if not qa_chain:
        st.warning("⚠️ No database found. Please upload a PDF to initialize the Retrieval system.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for src in message["sources"]:
                    st.write(src)

# React to user input
if prompt := st.chat_input("Ask a question about the document..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not qa_chain:
            st.error("Please upload a document to initialize the answering system.")
        else:
            with st.spinner("Searching for answers in the document..."):
                try:
                    response = qa_chain.invoke({'query': prompt})
                    answer = response["result"].strip()
                    
                    sources_extracted = []
                    if "source_documents" in response:
                        for idx, doc in enumerate(response["source_documents"]):
                            # Preview first 150 chars of the source document
                            content_preview = doc.page_content.replace('\n', ' ')[:150] + "..."
                            sources_extracted.append(f"**Chunk {idx+1}:** {content_preview}")
                    
                    st.markdown(answer)
                    if sources_extracted:
                        with st.expander("Show Retrieved Context"):
                            for src in sources_extracted:
                                st.write(src)
                                
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources_extracted
                    })
                except Exception as e:
                    st.error(f"Generative model failed: {e}")
