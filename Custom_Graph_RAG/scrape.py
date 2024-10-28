import os
import re
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Define paths for each disease's text files
output_dir_1 = r'C:\Users\Raghu\Downloads\private KEY\ETLpipeline\llm-vectors-unstructured\Custom_Graph_RAG\data\medical_texts\diabetes.txt'
output_dir_2 = r'C:\Users\Raghu\Downloads\private KEY\ETLpipeline\llm-vectors-unstructured\Custom_Graph_RAG\data\medical_texts\Alzhemier.txt'
output_dir_3 = r'C:\Users\Raghu\Downloads\private KEY\ETLpipeline\llm-vectors-unstructured\Custom_Graph_RAG\data\medical_texts\hypertension.txt'

# Function to extract disease information from the text
def extract_disease_info(text):
    patterns = {
        'disease_name': r'H1:\s*(.+)',
        'overview': r'H2:\s*Overview\s*(.*?)\n\n',
        'signs_and_symptoms': r'H2:\s*Signs and Symptoms\s*(.*?)\n\n',
        'causes': r'H2:\s*Causes\s*(.*?)\n\n',
        'diagnosis': r'H2:\s*Diagnosis and Tests\s*(.*?)\n\n',
        'treatment': r'H2:\s*Management and Treatment\s*(.*?)\n\n',
        'symptoms': r'H2:\s*Symptoms\s*(.*?)\n\n',  # Assuming there’s a section for symptoms
        'complications': r'H2:\s*Complications\s*(.*?)\n\n',  # Assuming there’s a section for complications
        'management': r'H2:\s*Management\s*(.*?)\n\n'  # Assuming there’s a section for management
    }
    
    extracted_info = {}
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted_info[key] = match.group(1).strip()
    
    return extracted_info

# Read text files with error handling for encoding
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as file:
            return file.read()

# Read disease texts
diabetes_text = read_text_file(output_dir_1)
alzheimer_text = read_text_file(output_dir_2)
hyper_tension_text = read_text_file(output_dir_3)

# Extract information for each disease
diabetes_info = extract_disease_info(diabetes_text)
alzheimer_info = extract_disease_info(alzheimer_text)
hyper_tension_info = extract_disease_info(hyper_tension_text)

# Initialize Hugging Face embeddings model
huggingface_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to load and process data for a given disease
def process_disease_data(file_path, disease_name):
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    # Initialize text splitter
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=1500,
        chunk_overlap=200,
    )

    # Split documents into chunks
    chunks = text_splitter.split_documents(docs)

    return chunks, disease_name

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
driver.verify_connectivity()

# Create a function to run the Cypher query
def create_disease_graph(tx, disease_name, chunks, extracted_info):
    for chunk in chunks:
        # Get the vector embedding for the chunk
        embedding = huggingface_emb.embed_query(chunk.page_content)
        
        data = {
            'text': chunk.page_content,
            'embedding': embedding,  # Embed the chunk content
            'condition': disease_name,
            'symptom': extracted_info.get('symptoms', ''),  # Get symptoms from extracted info
            'complication': extracted_info.get('complications', ''),  # Get complications from extracted info
            'management': extracted_info.get('management', ''),  # Get management info from extracted info
        }
        
        # Create nodes and relationships based on the disease type
        tx.run(""" 
            MERGE (d:Disease {name: $condition})
            MERGE (s:Symptom {name: $symptom})  // Adjust as needed for multiple symptoms
            MERGE (c:Complication {name: $complication})  // Adjust as needed for multiple complications
            MERGE (d)-[:HAS_SYMPTOM]->(s)
            MERGE (d)-[:HAS_COMPLICATION]->(c)
            MERGE (s)-[:MANAGED_BY]->(m:Management {name: $management})  // Adjust as needed for multiple management strategies
            WITH m
            CALL db.create.setNodeVectorProperty(m, "embedding", $embedding)  // Set embedding for management node
            RETURN COUNT(m)  // Optional: Return the count of nodes created/updated
        """, data)

# Process each disease
for output_dir, disease in [
    (output_dir_1, "Diabetes"), 
    (output_dir_2, "Alzheimer's Disease"), 
    (output_dir_3, "Hypertension")
]:
    chunks, disease_name = process_disease_data(output_dir, disease)
    
    # Extract information for the current disease
    extracted_info = None
    if disease == "Diabetes":
        extracted_info = diabetes_info
    elif disease == "Alzheimer's Disease":
        extracted_info = alzheimer_info
    elif disease == "Hypertension":
        extracted_info = hyper_tension_info

    with driver.session(database="neo4j") as session:
        session.execute_write(create_disease_graph, disease_name, chunks, extracted_info)

# Close the Neo4j driver connection
driver.close()

# Output the extracted information for each disease
for disease_name, disease_info in zip(["Diabetes", "Alzheimer's Disease", "Hypertension"], 
                                        [diabetes_info, alzheimer_info, hyper_tension_info]):
    print(f"{disease_name} Information:")
    for section, content in disease_info.items():
        print(f"{section.replace('_', ' ').title()}:\n{content}\n")
