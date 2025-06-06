# 🩺 Ask Your DigiDoctor 👨‍⚕️

An AI-powered health assistant that allows you to ask questions about **stroke** using information extracted from research papers. Built with [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), [HuggingFace Transformers](https://huggingface.co/), and [Groq’s LLaMA 3](https://groq.com/).

---

## 🚀 Features

- 🧠 Conversational AI using Groq's **LLaMA 3 70B** model
- 📚 Uploads and processes multiple PDFs using **LangChain document loaders**
- 📎 Stores document chunks in a **FAISS vector database**
- 🔍 Retrieves relevant answers using **semantic search**
- 💬 Remembers conversation history with **buffered memory**
- ⚡ Clean Streamlit interface for real-time Q&A

---

## 📁 File Structure
├── app.py # Main Streamlit app
├── data/ # Folder containing PDF documents
├── .env # Environment file for API keys
├── requirements.txt # Python dependencies
└── README.md # You're here! 

---

## 🧪 Sample Use Case

> You can ask:  
> - "What are the symptoms of stroke?"  
> - "How is stroke diagnosed?"  
> - "What treatment options exist?"  

The AI will search the uploaded stroke-related documents and provide relevant, conversational responses.

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/adubrightkwartengsnr/health-retrieval-augmented-generation.git
cd health-retrieval-augmented-generation

2. Create and activate a virtual environment
conda create -n rag-env python=3.10 -y
conda activate rag-env


3. Install dependencies
pip install -r requirements.txt

4. Add your API key
Create a .env file in the root directory:
GROQ_API_KEY=your_groq_api_key_here
🔐 Sign up for a Groq API key at https://console.groq.com/

5. Add your documents
Place relevant PDF files (e.g., stroke-related medical literature) in the data/ folder.

🏃 Run the App
streamlit run app.py
Open http://localhost:8501 in your browser.

🛠️ Tech Stack
Tool |	Purpose
Streamlit |	UI  for real-time Q&A
LangChain |	Memory and document QA chain
FAISS|	Fast vector similarity search
HuggingFace| Embeddings	Convert text into embeddings
Groq| LLaMA 3 API	Powerful LLM backend
PyPDFLoader|	Parse and load PDF documents


🧠 How It Works
PDFs are loaded and split into text chunks.
Chunks are embedded using sentence-transformers/all-MiniLM-L6-v2.
FAISS indexes the chunks for fast retrieval.
A conversational chain retrieves top chunks and queries the LLaMA model.
User conversation is tracked using ConversationBufferMemory.

🐛 Known Issues
If running on lower-spec machines, embeddings and FAISS may take time to load.
Warnings related to torch.classes during startup can be ignored (Streamlit internal behavior).
Replace .run() with .invoke() in the LangChain chain to avoid deprecation warnings.


📌 To-Do / Future Work
⬜ Enable document upload from the UI
⬜ Add multi-turn memory persistence (beyond session state)
⬜ Deploy on Streamlit Cloud or Hugging Face Spaces

📜 License
MIT License. Feel free to use, modify, and share.

🙋‍♂️ Contact
Developed by Bright Kwarteng Senior Adu
📧 [adubrightkwarrteng11@gmail.com]
🔗 https://www.linkedin.com/in/bright-adu-kwarteng-snr/

