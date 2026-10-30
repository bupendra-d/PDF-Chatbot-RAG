# PDF Chatbot

A simple PDF question-answering application built using Retrieval-Augmented Generation (RAG). Upload a PDF and ask questions based on its contents.

### Live Demo

[**Try the PDF Chatbot**](https://pdf-chatbot-rag-tvjl56z5stdjuq2gji5fs7.streamlit.app/)

### Features

- Upload and process PDF documents.
- Ask questions based on document contents.
- Retrieve relevant passages using semantic search.
- View source passages and page numbers.
- Generate answers using a Groq-hosted LLM.

### Tech Stack

Python · Streamlit · LangChain · FAISS · Hugging Face · Groq

### Run Locally

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd pdf-chatbot-rag
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure your API key**

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

**4. Run the application**

```bash
streamlit run app.py
```

### Deployment

Deploy the repository using [Streamlit Community Cloud](https://share.streamlit.io/) and add `GROQ_API_KEY` to your application's Secrets settings.

### Note

The chatbot uses retrieved passages from the uploaded PDF to generate answers. Responses may not always be accurate, so verify important information against the original document. Scanned PDFs may require OCR.

---
Developed using Python, LangChain, and Streamlit.