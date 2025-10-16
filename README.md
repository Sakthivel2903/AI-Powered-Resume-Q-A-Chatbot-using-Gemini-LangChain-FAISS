# AI-Powered-Resume-Q-A-Chatbot-using-Gemini-LangChain-FAISS
Here’s a **detailed and professional project description** for your code 👇

---

## 🤖 **AI-Powered Resume Q&A Chatbot using Gemini + LangChain + FAISS**

### 📋 **Project Description**

This project builds an **interactive Resume Chatbot** that allows users to **ask questions about a resume (PDF)** and get **concise, AI-generated answers** using **Google’s Gemini model** and **LangChain**.

It combines **PDF parsing**, **text embedding**, and **retrieval-based question answering (RAG)** to deliver accurate, short, and context-aware responses — ideal for HR automation, portfolio assistants, or AI interview preparation tools.

---

### 🚀 **Features**

* 📄 **Resume Parsing:** Automatically extracts and cleans text from any uploaded PDF resume.
* 🔍 **Semantic Search:** Uses **FAISS vector store** and **Gemini embeddings** for precise information retrieval.
* 🧠 **Conversational Memory:** Maintains chat history for contextual dialogue.
* ⚡ **Google Gemini Integration:** Uses **Gemini 2.5 Flash** for fast and reliable generative responses.
* 💬 **Concise Answers:** Responses are limited to 2–3 sentences for clarity and relevance.
* 🔒 **Environment Security:** Uses `.env` file to store sensitive API keys and file paths.

---

### 🧩 **How It Works**

1. The chatbot loads environment variables (`GOOGLE_API_KEY` and `RESUME_PDF_PATH`) from a `.env` file.
2. It extracts readable text from the PDF using **PyPDF**.
3. The text is split into small chunks using **LangChain’s RecursiveCharacterTextSplitter** for optimal retrieval.
4. Each chunk is embedded using **Google Generative AI Embeddings** and stored in a **FAISS vector database**.
5. When a user asks a question, the bot retrieves the most relevant chunks and sends them to **Gemini 2.5 Flash** for a short, human-like answer.
6. The chatbot continues the conversation with context retention, providing intelligent follow-ups.

---

### 🧠 **Technical Workflow**

```text
PDF File ➜ Text Extraction ➜ Text Chunking ➜ Embedding ➜ FAISS Vector Store ➜
Question Input ➜ Retriever (Top k=3) ➜ Gemini LLM (Short Answer Prompt) ➜ Response
```

---

### 📦 **Key Dependencies**

| Library                                  | Purpose                                          |
| ---------------------------------------- | ------------------------------------------------ |
| `pypdf`                                  | Extracts text from PDF resumes                   |
| `langchain`                              | Handles chunking, prompting, and chaining        |
| `langchain_community.vectorstores.FAISS` | Vector-based document retrieval                  |
| `langchain_google_genai`                 | Google Gemini integration for embeddings and LLM |
| `dotenv`                                 | Loads environment variables securely             |
| `FAISS`                                  | Efficient vector similarity search               |

---

### ⚙️ **Environment Setup**

1. **Install dependencies:**

   ```bash
   pip install python-dotenv pypdf langchain langchain-google-genai langchain-community faiss-cpu
   ```

2. **Create a `.env` file:**

   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   RESUME_PDF_PATH=path/to/your_resume.pdf
   ```

3. **Run the chatbot:**

   ```bash
   python resume_chatbot.py
   ```

---

### 💬 **Example Interaction**

```
============================================================
Resume Chatbot Ready. Ask questions or type 'quit' to exit.
============================================================

Your Question: What programming languages does Nithi know?

Thinking...

--- Answer ---
Nithi is proficient in Python, JavaScript, and Java, with additional experience in React and Node.js.

--- Source Documents Used ---
Chunk 1 (Page 1):
   -> Experienced in full-stack development using React, Node.js, and MongoDB...
============================================================
```

---

### 🎯 **Use Cases**

* 🧾 Automated Resume Review
* 💼 HR Screening Chatbot
* 🎓 Portfolio Assistant
* 🤖 Interview Preparation Tool
* 📚 AI Knowledge Base for PDF Documents

---

### ⚙️ **Code Structure Overview**

| Section                        | Function                                       |
| ------------------------------ | ---------------------------------------------- |
| `extract_text_from_pdf()`      | Reads and extracts text from a given PDF file. |
| `build_conversational_chain()` | Builds a Gemini + FAISS-powered Q&A pipeline.  |
| `main()`                       | Runs the interactive chatbot loop.             |

---

### 🧩 **Model Details**

* **Embedding Model:** `models/gemini-embedding-001`
* **LLM Model:** `gemini-2.5-flash`
* **Prompt Type:** Short-answer template (2–3 sentences max)
* **Vector Search Algorithm:** FAISS cosine similarity

---

### 🏁 **Conclusion**

This project is a **practical demonstration of Retrieval-Augmented Generation (RAG)** using Google’s Gemini models within the **LangChain** ecosystem.
It showcases how to combine AI reasoning with structured PDF data to create intelligent, domain-specific assistants for resumes, research papers, or documentation.

---

Would you like me to generate a **PDF project report** for this (like the earlier one) with diagrams and formatting?
