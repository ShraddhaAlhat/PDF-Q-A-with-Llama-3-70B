import os
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import faiss
import numpy as np
from groq import Groq

def extract_content_from_pdf(pdf_path):
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            documents.append(text)
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return text_splitter.create_documents(documents)

def get_embeddings(chunks):
    model = "text-embedding-004"
    texts = [chunk.page_content for chunk in chunks]
    embeddings = genai.embed_content(
        model=model,
        content=texts,
        task_type="retrieval_document"
    )
    return np.array(embeddings["embedding"]).astype('float32')

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_context(query, index, chunks, k=3):
    model = "text-embedding-004"
    result = genai.embed_content(
        model=model,
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = np.array([result["embedding"]]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    return [chunks[i].page_content for i in indices[0]]


def generate_response(query, context_chunks, groq_api_key):
    context_str = "\n\n".join(context_chunks)
    
    prompt = f"""Answer the question based only on the following context:
    
    {context_str}
    
    Question: {query}
    
    Provide a concise and accurate answer:"""
    
    # Initialize Groq client here
    groq_client = Groq(api_key=groq_api_key)
    
    chat_completion = groq_client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt
        }],
        model="llama3-70b-8192",
        temperature=0.3,
        max_tokens=1024
    )
    
    return chat_completion.choices[0].message.content
    


def fast_rag_pipeline(pdf_path, query, google_api_key, groq_api_key):
    # Configure APIs here
    genai.configure(api_key=google_api_key)
    
    # Process PDF
    documents = extract_content_from_pdf(pdf_path)
    
    # Split into chunks
    chunks = split_documents(documents)
    
    # Create embeddings and index
    embeddings = get_embeddings(chunks)
    index = create_faiss_index(embeddings)
    
    # Retrieve relevant context
    context_chunks = retrieve_context(query, index, chunks)
    
    # Generate final response - ADD groq_api_key HERE
    return generate_response(query, context_chunks, groq_api_key)