import gradio as gr
import os
from rag import fast_rag_pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process_pdfs(pdf_files, question):
    if not pdf_files:
        return "Please upload at least one PDF file"
    if not question:
        return "Please enter a question"
    
    if len(pdf_files) > 2:
        return "Please upload no more than 2 PDF files"
    
    try:
        # Combine all PDFs into a single processing pipeline
        pdf_paths = [pdf_file.name for pdf_file in pdf_files]
        
        # Get a single unified answer for all PDFs
        answer = fast_rag_pipeline(
            pdf_paths,  # Now accepts multiple files
            question,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        return answer
        
    except Exception as e:
        return f"Error processing PDFs: {str(e)}"
        
# Gradio interface
with gr.Blocks(title="PDF Q&A with Llama 3 70B") as demo:
    gr.Markdown("# 📄 PDF Q&A with Llama 3 70B")
    gr.Markdown("Upload up to 2 PDF files and ask questions")
    
    with gr.Row():
        pdf_input = gr.File(label="Upload PDF files", file_types=[".pdf"], file_count="multiple")
        question_input = gr.Textbox(label="Ask your question", placeholder="What would you like to know about these documents?", lines=5)
    
    submit_btn = gr.Button("Get Answer")
    output = gr.Textbox(label="Answer", interactive=False)
    
    submit_btn.click(
        fn=process_pdfs,
        inputs=[pdf_input, question_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()