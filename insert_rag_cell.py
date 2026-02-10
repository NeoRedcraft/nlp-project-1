
import json
import os

notebook_path = r'c:\Users\Felipe III\OneDrive - Mapúa University\Documents\GitHub\nlp-project-1\Final.ipynb'

# New code content for the RAG preprocessing cell
new_source = [
    "# RAG Preprocessing for Qwen/Qwen3 Model\n",
    "from langchain_community.document_loaders import CSVLoader\n",
    "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
    "from langchain_huggingface import HuggingFaceEmbeddings\n",
    "from langchain_community.vectorstores import Chroma\n",
    "\n",
    "# 1. Load Data\n",
    "loader = CSVLoader(file_path='dataset/Mental_Health_FAQ.csv', source_column='Questions', encoding='utf-8')\n",
    "documents = loader.load()\n",
    "\n",
    "# 2. Text Splitting\n",
    "text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)\n",
    "splits = text_splitter.split_documents(documents)\n",
    "\n",
    "# 3. Embedding Model (Preparing for Qwen Retrieval)\n",
    "embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'\n",
    "embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)\n",
    "\n",
    "# 4. Vector Store Creation\n",
    "persist_directory = './chroma_db'\n",
    "vectorstore = Chroma.from_documents(\n",
    "    documents=splits,\n",
    "    embedding=embeddings,\n",
    "    persist_directory=persist_directory\n",
    ")\n",
    "print(f'Vector store created successfully at {persist_directory} with {len(splits)} chunks.')"
]

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the index of the markdown cell for Section 2.5
    target_id = 'fTlYQlly0yUM'
    insert_index = -1
    
    for i, cell in enumerate(nb['cells']):
        if cell.get('metadata', {}).get('id') == target_id:
            insert_index = i + 1
            break
    
    if insert_index != -1:
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": "rag_preprocessing_code"}, # New ID
            "outputs": [],
            "source": new_source
        }
        
        nb['cells'].insert(insert_index, new_cell)
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
        print("Successfully inserted RAG preprocessing cell.")
    else:
        print(f"Target cell ID '{target_id}' not found.")

except Exception as e:
    print(f"Error: {e}")
