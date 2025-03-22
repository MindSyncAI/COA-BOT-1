from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import pickle

def generate_embeddings():
    print("Starting embedding generation...")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': "cpu"}
    )
    
    documents = []
    pdf_files = []
    
    # Find all PDF files in the dataset directory
    for root, _, files in os.walk('dataset'):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print("No PDF files found in the dataset directory!")
        return
    
    # Load each PDF file individually
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
            print(f"Successfully loaded {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    if not documents:
        print("No documents were successfully loaded!")
        return
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    
    # Save the vector store
    vector_store.save_local("embeddings")
    print("Embeddings saved successfully to 'embeddings' directory")
    
    # Save metadata about the documents
    metadata = {
        "num_documents": len(documents),
        "num_chunks": len(text_chunks),
        "pdf_files": pdf_files
    }
    
    with open("embeddings/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print("Metadata saved successfully")

if __name__ == "__main__":
    generate_embeddings() 