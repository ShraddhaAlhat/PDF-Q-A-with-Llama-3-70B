import streamlit as st
import os
from rag_pipeline import fast_rag_pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="PDF Q&A with Llama 3",
    page_icon="📄",
    layout="wide"
)

# Main interface
st.title("📄 PDF Q&A with Llama 3 70B")
st.markdown("Upload up to 2 PDF files and ask questions")

# File uploader - limit to 2 files
uploaded_files = st.file_uploader(
    "Choose PDF files (max 2)",
    type="pdf",
    accept_multiple_files=True
)

# Limit to 2 files
if uploaded_files and len(uploaded_files) > 2:
    st.error("Please upload no more than 2 PDF files")
    st.stop()

if uploaded_files:
    # Display file info
    for file in uploaded_files:
        st.success(f"Uploaded: {file.name}")
    
    # Question input
    question = st.text_area(
        "Ask your question:",
        height=100,
        placeholder="What would you like to know about these documents?"
    )
    
    if st.button("Get Answer") and question:
        with st.spinner("Analyzing documents and generating answer..."):
            try:
                # Process all PDFs and combine content
                all_context = []
                for file in uploaded_files:
                    # Save temp file
                    temp_pdf = os.path.join("temp", file.name)
                    os.makedirs("temp", exist_ok=True)
                    with open(temp_pdf, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Get answer for each PDF (or combine processing)
                    answer = fast_rag_pipeline(
                        temp_pdf, 
                        question,
                        google_api_key=os.getenv("GOOGLE_API_KEY"),
                        groq_api_key=os.getenv("GROQ_API_KEY")
                    )
                    all_context.append(answer)
                    
                    # Clean up
                    os.remove(temp_pdf)
                
                # Combine answers or show separately
                st.subheader("Answer:")
                for i, answer in enumerate(all_context, 1):
                    st.markdown(f"**From PDF {i}:**")
                    st.markdown(answer)
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")