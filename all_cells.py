
# --- Cell 6 ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
file_path = 'Mental_Health_FAQ.csv' # Changed path to reflect common Colab upload location
df = pd.read_csv(file_path)

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['Questions'].dropna())

# Get feature names and sum TF-IDF scores
feature_names = vectorizer.get_feature_names_out()
dense = tfidf_matrix.todense()
denselist = dense.tolist()
df_tfidf = pd.DataFrame(denselist, columns=feature_names)
top_n = 20
tfidf_sum = df_tfidf.sum().sort_values(ascending=False).head(top_n)

# Plot TF-IDF Bar Chart
plt.figure(figsize=(12, 6))
sns.barplot(x=tfidf_sum.values, y=tfidf_sum.index, palette='viridis')
plt.title(f'Top {top_n} Words in Questions by TF-IDF Score')
plt.xlabel('TF-IDF Score')
plt.ylabel('Words')
plt.show()

# --- Cell 8 ---
# Clone the GitHub repository
!git clone https://github.com/NeoRedcraft/nlp-project-1

# --- Cell 9 ---
%cd nlp-project-1

# --- Cell 10 ---
!pip install -r requirements.txt

# --- Cell 11 ---
!pip install --quiet langchain-huggingface

# --- Cell 12 ---
!pip install --quiet transformers accelerate bitsandbytes langchain-huggingface

# --- Cell 13 ---
# RAG Preprocessing for Qwen/Qwen1.5-7B Model
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # Updated import path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document # Updated import path
import os

def main():
    # 1. Load Data
    # Ensure the path is correct relative to where you run this script
    file_path = '/content/nlp-project-1/dataset/Mental_Health_FAQ.csv' # Updated path to refer to the cloned GitHub repo
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

# --- Cell 14 ---
import os
from google.colab import files

# Define the path to the chroma_db folder
chroma_db_path = './chroma_db'
zip_file_name = 'chroma_db.zip'

# Ensure we are in the base content directory before zipping
# The chroma_db is created in the current working directory, which should be /content/nlp-project-1.
# If you need to access it from /content, you might need to adjust paths or chdir.
# For now, assuming current directory has chroma_db.

# Compress the chroma_db folder
if os.path.exists(chroma_db_path):
    # Using -r for recursive, -q for quiet (no output)
    !zip -r -q {zip_file_name} {chroma_db_path}
    print(f"'{chroma_db_path}' successfully zipped as '{zip_file_name}'.")

    # Download the zip file
    files.download(zip_file_name)
    print(f"'{zip_file_name}' download initiated.")
else:
    print(f"Error: The folder '{chroma_db_path}' does not exist. Please ensure it has been created successfully.")


# --- Cell 15 ---
!pip install langchain-classic

# --- Cell 16 ---
!pip install -U bitsandbytes

# --- Cell 17 ---
!pip install langchain langchain-text-splitters langchain-community bs4

# --- Cell 18 ---
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig # Import BitsAndBytesConfig
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
# The original import was from langchain_classic.chains, but RetrievalQA is typically from langchain.chains
from langchain_classic.chains import RetrievalQA # Corrected import for RetrievalQA

# Install necessary libraries if not already installed
# Note: accelerate is for efficient model loading, bitsandbytes for quantization
# Re-running pip install to ensure latest versions are available, particularly for bitsandbytes and transformers
!pip install --quiet --upgrade bitsandbytes accelerate transformers huggingface_hub langchain-huggingface
# Force reinstall langchain and langchain-community, ensuring all dependencies are handled
!pip install --quiet --force-reinstall langchain langchain-community

# 1. Define the Qwen model ID
model_id = "Qwen/Qwen1.5-7B-Chat"

# 2. Load Tokenizer and Model (with quantization for memory efficiency)
print(f"Loading tokenizer and model for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define quantization configuration using BitsAndBytesConfig
# This addresses the warning and potential compatibility issues with load_in_4bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config, # Pass the quantization_config object
    device_map="auto", # Automatically maps model layers to available devices (CPU/GPU)
    # The torch_dtype here will apply to non-quantized parts or if 4-bit is not used.
    # For 4-bit, bnb_4bit_compute_dtype is more relevant.
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
)

# 3. Create a Hugging Face text generation pipeline
print("Creating Hugging Face text generation pipeline...")
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512, # Maximum number of tokens to generate
    do_sample=True,     # Use sampling for generation
    temperature=0.7,    # Sampling temperature
    top_k=50,           # Top-k sampling
    top_p=0.95,         # Top-p (nucleus) sampling
)

# 4. Initialize HuggingFacePipeline for LangChain
llm = HuggingFacePipeline(pipeline=pipeline)
print("HuggingFace LLM initialized for LangChain.")

# 5. Load the embedding model (same as used for Chroma DB creation)
print("Initializing embedding model...")
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# 6. Load the Chroma vector store
persist_directory = './chroma_db'
print(f"Loading Chroma vector store from {persist_directory}...")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# 7. Create a retriever from the vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant documents
print("Retriever created.")

# 8. Set up the RAG chain using RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "stuff" combines all documents into one prompt
    retriever=retriever,
    return_source_documents=True
)
print("RetrievalQA chain initialized with Qwen1.5-7B and Chroma DB.")

print("RAG system is ready. You can now use 'qa_chain.invoke({'query': 'Your question here'})' to query the system.")

# --- Cell 54 ---
combined_text = df[question_col].astype(str) + " " + df[answer_col].astype(str)
combined_text = combined_text.fillna('')

print("Combined text series created with {} entries.".format(len(combined_text)))
print("First 5 entries of combined_text:\n", combined_text.head())

# --- Cell 57 ---
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer with English stop words
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Apply fit_transform to the combined_text series
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)

print("TF-IDF matrix created with shape:", tfidf_matrix.shape)

# --- Cell 60 ---
import pandas as pd

# 1. Extract feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# 2. Calculate the sum of TF-IDF scores for each term
# tfidf_matrix is a sparse matrix, sum(axis=0) sums columns
term_tfidf_sums = tfidf_matrix.sum(axis=0).A1

# 3. Create a Pandas Series mapping feature names to their summed TF-IDF scores
tfidf_scores = pd.Series(term_tfidf_sums, index=feature_names)

# 4. Sort the Series in descending order
sorted_tfidf_scores = tfidf_scores.sort_values(ascending=False)

# 5. Select the top N terms (e.g., top 20)
top_n = 20
top_tfidf_terms = sorted_tfidf_scores.head(top_n)

print(f"Top {top_n} TF-IDF terms and their scores:")
print(top_tfidf_terms)

# --- Cell 66 ---
import matplotlib.pyplot as plt
import seaborn as sns

# Create a bar plot
plt.figure(figsize=(10, 8))
sns.barplot(x=top_tfidf_terms.values, y=top_tfidf_terms.index, palette='viridis')

# Set title and labels
plt.title('Top TF-IDF Terms')
plt.xlabel('TF-IDF Score')
plt.ylabel('Terms')

# Ensure all elements fit within the figure
plt.tight_layout()

# Display the plot
plt.show()


# --- Cell 68 ---
import matplotlib.pyplot as plt
import seaborn as sns

# Create a bar plot
plt.figure(figsize=(10, 8))
sns.barplot(x=top_tfidf_terms.values, y=top_tfidf_terms.index, legend=False)

# Set title and labels
plt.title('Top TF-IDF Terms')
plt.xlabel('TF-IDF Score')
plt.ylabel('Terms')

# Ensure all elements fit within the figure
plt.tight_layout()

# Display the plot
plt.show()

# --- Cell 71 ---
import re
import string

# 1. Convert to lowercase
cleaned_text = combined_text.str.lower()

# 2. Remove punctuation
cleaned_text = cleaned_text.apply(lambda x: re.sub(f'[{re.escape(string.punctuation)}]', '', x))

# 3. Remove numerical digits
cleaned_text = cleaned_text.apply(lambda x: re.sub(r'\d+', '', x))

# 4. Remove extra whitespace
cleaned_text = cleaned_text.apply(lambda x: re.sub(r'\s+', ' ', x).strip())

print("Text cleaning complete. First 5 entries of cleaned_text:")
print(cleaned_text.head())

# --- Cell 74 ---
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the RecursiveCharacterTextSplitter
chunk_size = 500
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    add_start_index=True,
)

# Convert the cleaned_text Series to a list of strings for the splitter
texts_to_chunk = cleaned_text.tolist()

# Apply the text splitter to the cleaned text
chunks = text_splitter.create_documents(texts_to_chunk)

print(f"Number of chunks created: {len(chunks)}")
print("First 5 chunks:")
for i, chunk in enumerate(chunks[:5]):
    print(f"Chunk {i+1} (start index {chunk.metadata['start_index']}): {chunk.page_content[:200]}...")

# --- Cell 76 ---
import sys
!pip install --quiet langchain

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the RecursiveCharacterTextSplitter
chunk_size = 500
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    add_start_index=True,
)

# Convert the cleaned_text Series to a list of strings for the splitter
texts_to_chunk = cleaned_text.tolist()

# Apply the text splitter to the cleaned text
chunks = text_splitter.create_documents(texts_to_chunk)

print(f"Number of chunks created: {len(chunks)}")
print("First 5 chunks:")
for i, chunk in enumerate(chunks[:5]):
    print(f"Chunk {i+1} (start index {chunk.metadata['start_index']}): {chunk.page_content[:200]}...")

# --- Cell 78 ---
import sys
!pip install --quiet langchain==0.1.16

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the RecursiveCharacterTextSplitter
chunk_size = 500
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    add_start_index=True,
)

# Convert the cleaned_text Series to a list of strings for the splitter
texts_to_chunk = cleaned_text.tolist()

# Apply the text splitter to the cleaned text
chunks = text_splitter.create_documents(texts_to_chunk)

print(f"Number of chunks created: {len(chunks)}")
print("First 5 chunks:")
for i, chunk in enumerate(chunks[:5]):
    print(f"Chunk {i+1} (start index {chunk.metadata['start_index']}): {chunk.page_content[:200]}...")

# --- Cell 83 ---


# --- Cell 85 ---


# --- Cell 87 ---


# --- Cell 89 ---


# --- Cell 91 ---


# --- Cell 93 ---


# --- Cell 95 ---


# --- Cell 97 ---


# --- Cell 99 ---


# --- Cell 101 ---


# --- Cell 103 ---


# --- Cell 105 ---


# --- Cell 107 ---


# --- Cell 109 ---

