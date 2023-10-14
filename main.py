from flask import Flask, request, jsonify
from flask_cors import CORS  # Import the CORS class

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
app = Flask(__name__)
CORS(app)  # Initialize CORS with your Flask app
APIKEY = "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"

os.environ["OPENAI_API_KEY"] = APIKEY

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,    
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


pdf_files_list = [
    r'data/Operator User Manual Diabos 3.0 (DA-Stolt_Operations).pdf',
    r'data/Operator User Manual Diabos 3.0 (DA-Stolt_Accounts).pdf',
    r'data/Operator User Manual Diabos 3.0 (DA-Odjfell).pdf',
    r'data/Operator User Manual Diabos 3.0 (DA-CMS) (20Mar23) New Version.pdf',
    r'data/Operator User Manual Diabos 3.0 (20Mar23) New Version.pdf',
    r'data/Diabos User Manual Diabos 3.0 (DA) (19Apr23).pdf',
    r'data/AgentUser Manual Diabos 3.0 (DA) (04May23).pdf'
]

raw_text = get_pdf_text(pdf_files_list)
text_chunks = get_text_chunks(raw_text)
vectorstore = get_vectorstore(text_chunks)
chain = get_conversation_chain(vectorstore)

chat_history = []

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    if 'input_field' in data:
        input_value = data['input_field']
    query = input_value.lower()
    if query=='exit':
        result = {"answer": "It was nice talking to you. If you have any other questions, please feel free to ask me."}         
    else:
        result = chain({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    return jsonify({'content': result['answer']})

if __name__ == '__main__':
    app.run(debug=True)