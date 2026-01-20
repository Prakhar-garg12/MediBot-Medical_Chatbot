from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore 
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medical"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding = embeddings
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k":3})

llm = ChatOllama(
    model="gemma:2b",
    temperature=0,
    
)



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

rag_chain = (
    {
        "context": retriever,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

def clean_answer(text: str) -> str:
    if "\n\n" in text:
        return text.split("\n\n", 1)[1].strip()
    return text.strip()



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke(msg)
    response = clean_answer(response)
    print("Response : ", response)
    return response

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080, debug= True)