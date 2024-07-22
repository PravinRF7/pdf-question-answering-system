import os
import logging
import matplotlib.pyplot as plt
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import textract
from transformers import GPT2TokenizerFast
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

local_path = "Corpus.pdf"

# Function to handle PDF loading and splitting
def load_and_split_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        logger.info("PDF loaded and split successfully.")
        return pages
    except Exception as e:
        logger.error(f"Error loading or splitting PDF: {e}")
        st.error(f"Error loading or splitting PDF: {e}")
        return []

# Function to handle text extraction
def extract_text(pdf_path):
    try:
        doc = textract.process(pdf_path)
        logger.info("Text extracted from PDF successfully.")
        return doc.decode('utf-8')
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to count tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Load and split the PDF
pages = load_and_split_pdf(local_path)
chunks = pages

# Extract text and save to file
text = extract_text(local_path)
with open('corpus_1.txt', 'w') as f:
    f.write(text)

# Load text from file
with open('corpus_1.txt', 'r') as f:
    text = f.read()

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=24,
    length_function=count_tokens,
)

chunks = text_splitter.create_documents([text])

# Create a vector database
try:
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag"
    )
    logger.info("Vector database created successfully.")
except Exception as e:
    logger.error(f"Error creating vector database: {e}")
    st.error(f"Error creating vector database: {e}")

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

# LLM from Ollama
local_model = "mistral"
llm = ChatOllama(model=local_model)

# Query prompt
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Retriever
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm,
    prompt=QUERY_PROMPT
)

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask_question(question):
    try:
        result = chain.invoke(question)
        return result
    except Exception as e:
        logger.error(f"Error during query processing: {e}")
        st.error(f"Error during query processing: {e}")
        return "An error occurred while processing your query. Please try again."

# Streamlit Interface
st.title("PDF Question Answering System")
st.write("Ask questions based on the content of the provided PDF document.")

user_question = st.text_input("Enter your question:")
if user_question:
    response = ask_question(user_question)
    st.write("**Answer:**")
    st.write(response)
