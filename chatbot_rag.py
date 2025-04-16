from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
import asyncio
from argostranslate import package, translate
from langdetect import detect

# Initialize FastAPI
app = FastAPI()

# Add CORS Middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load documents
loader = TextLoader(file_path='page_content.txt', encoding='utf-8')
docs = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Embed documents for normal method
MODEL = 'phi3.5:3.8b-mini-instruct-q8_0'
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=OllamaEmbeddings(model='nomic-embed-text')
)

# Setup retriever
retriever = vectorstore.as_retriever()

# Load prompt for normal method
prompt = hub.pull("rlm/rag-prompt")

# Setup LLM
llm = ChatOllama(model=MODEL, temperature=0)

# Post-processing function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Setup RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Define async generator function to stream the response
async def stream_response(query):
    async for chunk in rag_chain.astream(query):
        yield chunk

# Expose a GET API for questions
@app.get("/")
async def welcome():
    return {"message": "Welcome to the Department of Justice Query API!"}

# Define a Pydantic model for the request body
class QueryRequest(BaseModel):
    query: str

# Initialize translation
def initialize_translation(from_code, to_code):
    package.update_package_index()
    available_packages = package.get_available_packages()
    
    # Filter and install the package
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, 
            available_packages
        ),
        None  # Handle case if no package is found
    )
    
    if package_to_install:
        package.install_from_path(package_to_install.download())
    else:
        raise Exception(f"No translation package found for {from_code} to {to_code}")

# Detect if the text is in Hindi using langdetect
def is_hindi(text):
    try:
        return detect(text) == 'hi'
    except:
        return False

# Translate text
def translate_text(text, from_code, to_code):
    translation = translate.get_translation_from_codes(from_code, to_code)
    
    if translation is None:
        raise Exception(f"No translation found from {from_code} to {to_code}")
    
    return translation.translate(text)

# Initialize both Hindi -> English and English -> Hindi translations
initialize_translation("hi", "en")
initialize_translation("en", "hi")

# Process input with translation if necessary
def process_input(text):
    input_in_hindi = is_hindi(text)
    
    # Translate Hindi input to English if necessary
    if input_in_hindi:
        text = translate_text(text, "hi", "en")
    
    # Now process the query as normal (only English text is passed further)
    output_text = text
    
    # If the input was in Hindi, translate the output back to Hindi
    if input_in_hindi:
        output_text = translate_text(output_text, "en", "hi")
    
    # Return only the processed English response, no original Hindi part
    return output_text

# Expose a POST API to accept queries and stream responses
@app.post("/query/")
async def get_query_result(request: QueryRequest):
    query = request.query
    processed_query = process_input(query)  # Process the query with translation
    return StreamingResponse(stream_response(processed_query), media_type="text/plain")
