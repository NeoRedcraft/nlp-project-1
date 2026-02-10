# RAG Preprocessing for Qwen/Qwen3 Model
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os

def main():
    # 1. Load Data
    # Ensure the path is correct relative to where you run this script
    file_path = 'Mental_Health_FAQ.csv'
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Create a new column that combines Question and Answer for the embedding content
    # This ensures the retrieval system finds relevant answers based on query similarity
    # to both the question and the answer content.
    df['combined_content'] = 'Question: ' + df['Questions'] + '\nAnswer: ' + df['Answers']

    # 2. Create LangChain Documents
    print("Creating documents...")
    documents = []
    for index, row in df.iterrows():
        # strict handling of potential NaN values
        content = str(row['combined_content']) if pd.notna(row['combined_content']) else ""
        doc = Document(
            page_content=content,
            metadata={'Question_ID': row['Question_ID']}
        )
        documents.append(doc)

    # 3. Text Splitting
    # Splitting might be needed if some answers are very long.
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # 4. Embedding Model (Preparing for Qwen Retrieval)
    # sentences-transformers/all-MiniLM-L6-v2 is a good general purpose embedding model
    print("Initializing embedding model...")
    embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # 5. Vector Store Creation
    persist_directory = './chroma_db'
    print(f"Creating vector store at {persist_directory}...")
    
    # Optional: Clear existing DB to avoid duplicates if running multiple times
    # import shutil
    # if os.path.exists(persist_directory):
    #     shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector store created successfully at {persist_directory} with {len(splits)} documents.")

if __name__ == "__main__":
    main()
