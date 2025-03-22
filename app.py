from flask import Flask, render_template, request, jsonify, session
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

def create_chain():
    # Get API key
    groq_api_key = os.environ.get("GROQ_API_KEY")
    
    # Load pre-computed embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load the vector store from disk with deserialization allowed
    vector_store = FAISS.load_local(
        "embeddings", 
        embeddings,
        allow_dangerous_deserialization=True  # We trust our own embeddings
    )
    
    # Load metadata
    with open("embeddings/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Loaded {metadata['num_documents']} documents with {metadata['num_chunks']} chunks")
    
    # Initialize LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0.2,
        max_tokens=4000
    )
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create conversation chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    
    return chain

@app.route('/')
def home():
    # Reset session when user visits the home page
    session.clear()
    session['conversation_history'] = []
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'answer': 'Please ask a question'})
    
    # Initialize conversation history if it doesn't exist
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    
    # Create a new chain for this request
    chain = create_chain()
    
    # Get response from the chain
    result = chain({"question": user_message, "chat_history": session['conversation_history']})
    answer = result["answer"]
    
    # Update conversation history in session
    session['conversation_history'].append((user_message, answer))
    
    return jsonify({'answer': answer})

@app.route('/reset', methods=['POST'])
def reset():
    session['conversation_history'] = []
    return jsonify({'status': 'success', 'message': 'Conversation reset successfully'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
