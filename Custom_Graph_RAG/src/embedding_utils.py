import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama  # Ollama LLM integration
from langchain.chains import RetrievalQA
import streamlit as st  # Import Streamlit

# Load environment variables
load_dotenv()

# Define paths for each disease's text files
output_dir_1 = r'C:\Users\Raghu\Downloads\private KEY\ETLpipeline\llm-vectors-unstructured\Custom_Graph_RAG\data\medical_texts\diabetes.txt'
output_dir_2 = r'C:\Users\Raghu\Downloads\private KEY\ETLpipeline\llm-vectors-unstructured\Custom_Graph_RAG\data\medical_texts\Alzhemier.txt'
output_dir_3 = r'C:\Users\Raghu\Downloads\private KEY\ETLpipeline\llm-vectors-unstructured\Custom_Graph_RAG\data\medical_texts\hypertension.txt'

# Function to read text file
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()

# Read the texts
diabetes_text = read_text_file(output_dir_1)
alzheimer_text = read_text_file(output_dir_2)
hypertension_text = read_text_file(output_dir_3)

# Define a simple Document class
class Document:
    def __init__(self, content):
        self.page_content = content
        self.metadata = {}  # Initialize metadata as an empty dictionary

# Initialize RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# Create document-like structures and split into chunks
diabetes_doc = Document(diabetes_text)
alzheimer_doc = Document(alzheimer_text)
hypertension_doc = Document(hypertension_text)
diabetes_chunks = text_splitter.split_documents([diabetes_doc])
alzheimer_chunks = text_splitter.split_documents([alzheimer_doc])
hypertension_chunks = text_splitter.split_documents([hypertension_doc])

# Combine all document chunks
docs = diabetes_chunks + alzheimer_chunks + hypertension_chunks

# Embedding model setup using all-MiniLM-L6-v2
modelPath = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Create FAISS index with embedded documents
db = FAISS.from_documents(docs, embedding=embeddings)

# Set up the Ollama LLM model
llm = Ollama(model="llama3.1")  # or use other supported models like "mistral"

# Define the retriever
retriever = db.as_retriever(search_kwargs={"k": 5})  # Retrieve more docs for richer context

# Define a RetrievalQA using the Ollama LLM
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Streamlit UI
st.title("Medical Condition Q&A")
st.write("Ask about symptoms and information related to Diabetes, Alzheimer's, and Hypertension.")

# Input question from the user
question = st.text_input("Enter your question:")

if question:
    # Get the answer from the QA system
    result = qa.run({"query": question})
    
    # Display the result
    st.write("### Answer:")
    st.write(result)
