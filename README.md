# 📚 PDF Chat with Google Gemini (RAG)

A modular Streamlit application that allows you to **upload multiple PDFs**, generate embeddings with **Google Generative AI**, and ask questions using a **Retrieval-Augmented Generation (RAG)** chain.  

The project is structured for **modularity, scalability, and maintainability**.
---

## Features

- Upload multiple PDFs and extract text automatically  
- Split PDF text into chunks for better retrieval  
- Generate embeddings using Google Gemini  
- Store embeddings in **FAISS** local vector store  
- Ask questions and get detailed answers from uploaded PDFs  
- Modular code structure for easy maintenance  
- Optional Docker support for deployment 


## Technologies Used:
- Streamlit, Python
- Models used: Google-Gemini-Pro-2.5
- Langchain
- Vector Database (FAISS, Chromadb)

## Project Structure

```bash
Gemini_App/
├── app.py # Streamlit app
├── pdf_utils.py # Streamlit app
├── rag_chain.py
├── vecotrstores_utils.py
├── secrets.env # To store the api key
├── requirements.txt # Python dependencies
├── Dockerfile # Optional Docker container
├── README.md
├── run.sh # Optional to run the script
└── setup.py # Optional
```

## Files use in detail
```bash
- pdf_utils.py --> PDF reading + splitting
- vecotrstores_utils.py --> FAISS embeddings + storage
- rag_chain.py --> RAG Chain + prompt + ChatGoogleGenerativeAI
- app.py --> Streamlit interface
```

## Installation

## 🛠 Installation (without Docker)

### 1. Clone the repo
```bash
git clone https://github.com/AmreetNanda/Multi_PDF_Chat_Gemini_Pro.git
cd Gemini_App
```
### 2. Requirements.txt
```bash
streamlit
google-generativeai
python-dotenv
PyPDF2
Chromadb
faiss-cpu
langchain
langchain-core
langchain-google-genai
langchain-text-splitters
langchain_community
langchain-classic
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create .env file
```bash
Add the google api key here 
GOOGLE_API_KEY= "paste the api_key here"
```

### 5. Run Streamlit app
```bash
streamlit run app.py
```
Open in your browser:
```
👉 http://127.0.0.1:5000/
👉 Click the "Predict Disease" button.
👉 Receive the predicted skin disease and confidence level.
```

## 🐳 Running with Docker (optional)
### Build the image
```bash
docker build -t pdf-gemini-app .
```

### Run the container
```bash
docker run -p 8501:8501 -v $(pwd)/Models:/app/Models -v $(pwd)/.env:/app/.env pdf-gemini-app
```
```
- Mount your Model/ folder if using local Model
- Mount your .env file to provide the Google API key
- Open: 👉 http://localhost:8501
```
## License

[MIT](https://choosealicense.com/licenses/mit/)