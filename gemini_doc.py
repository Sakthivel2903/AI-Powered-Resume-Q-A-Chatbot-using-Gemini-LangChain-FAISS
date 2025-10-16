import os
from dotenv import load_dotenv
from pathlib import Path
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Tuple

# --- Load Environment Variables ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_PATH_STR = os.getenv("RESUME_PDF_PATH")

if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in your .env file.")
if not PDF_PATH_STR:
    raise ValueError("Set RESUME_PDF_PATH in your .env file.")

PDF_PATH = Path(PDF_PATH_STR).expanduser().resolve()

# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using pypdf"""
    print(f"1. Extracting text from PDF: {pdf_path}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    if not text.strip():
        raise ValueError("The PDF is empty or text extraction failed.")
        
    return text

# --- Build Conversational Chain ---
def build_conversational_chain(pdf_path: Path, api_key: str):
    """
    Builds a conversational retrieval QA chain using Gemini and FAISS,
    with a prompt that forces short, concise answers.
    """
    resume_text = extract_text_from_pdf(pdf_path)

    # Split text into chunks for retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = splitter.split_text(resume_text)
    print("2. Splitting text into chunks...")
    print(f"   -> Created {len(texts)} text chunks.")
    
    print("3. Initializing Gemini Embeddings and FAISS Vector Store...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    vectorstore = FAISS.from_texts(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("4. Initializing Gemini Generative Model...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        google_api_key=api_key,
        max_output_tokens=1500 
    )

    print("5. Creating Short-Answer Prompt...")
    short_answer_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Based on the following context, "
            "provide a short, concise answer to the question. "
            "Answer in simple language and keep it within 2–3 sentences.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

    # Build chain with retriever and short-answer prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": short_answer_prompt}  # ensure short answer
    )
    print("--- Setup Complete ---")
    return qa_chain

# --- Main Chatbot Loop ---
def main():
    try:
        qa_chain = build_conversational_chain(PDF_PATH, GOOGLE_API_KEY)
    except Exception as e:
        print(f"\nFATAL SETUP ERROR: {e}")
        print("Please check your API key, PDF path, and network connectivity.")
        return
        
    chat_history: List[Tuple[str, str]] = [] 

    print("\n============================================================")
    print("Resume Chatbot Ready. Ask questions or type 'quit' to exit.")
    print("============================================================")

    while True:
        try:
            question = input("\nYour Question: ")
            if question.lower() in ("quit", "exit"):
                print("Exiting...")
                break
            if not question.strip():
                continue

            print("\nThinking...")
            result = qa_chain.invoke({"question": question, "chat_history": chat_history})

            # Print the short answer safely
            answer_text = result.get("answer") or result.get("result") or result.get("output_text") or "⚠️ No answer returned"
            print("\n--- Answer ---")
            print(answer_text)

            # Show which source chunks were used
            print("\n--- Source Documents Used ---")
            for i, doc in enumerate(result["source_documents"]):
                snippet = doc.page_content.replace("\n", " ")
                page_info = doc.metadata.get('page', 'N/A')
                print(f"Chunk {i+1} (Page {page_info}):")
                print(f"   -> {snippet[:150]}...")
            print("=" * 60)

            # Update chat history
            chat_history.append((question, answer_text))

        except Exception as e:
            print(f"\nAn error occurred during query processing: {e}")
            break

if __name__ == "__main__":
    main()

