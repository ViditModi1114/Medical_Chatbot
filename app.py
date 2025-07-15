from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize Groq client with API key from environment variable
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY is required")
client = Groq(api_key=API_KEY)

# Global variable for vectorstore
vectorstore = None

# Function to detect and translate text
def detect_and_translate(text, target_lang="en"):
    translator = GoogleTranslator(source='auto', target=target_lang)
    return translator.translate(text)

# Extracting text from PDF
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Converting text to chunks
def get_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(raw_text)
    return chunks

# Using SentenceTransformers embeddings and FAISS to get vectorstore
def get_vectorstore(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# Create a class that implements Runnable for the Groq API
class GroqChat:
    def __init__(self, model_id):
        self.model_id = model_id
        self.client = Groq()

    def __call__(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""
        return response

# Function to handle the user's question and generate a response using Groq
def handle_question(question, vectorstore, user_lang):
    # Translate the question to English before sending it to the model
    question_translated = detect_and_translate(question, target_lang="en")

    # Find the closest text chunk from the vectorstore if it exists
    if vectorstore:
        retrieved_chunk = vectorstore.similarity_search(question_translated)
        context = retrieved_chunk[0].page_content if retrieved_chunk else "No relevant information found."
    else:
        context = "This is a general response as no PDF content is available."

    # Create a message to send to the Groq model
    prompt = f"""

You are a healthcare assistant. Answer the following user question with a detailed and structured response. The response should include the following points if relevant to the query. Adjust the points dynamically based on the question's context. Use plain and easy-to-understand language.

Relevant Points:
- Home Remedies (if applicable)
- Medication Advice (if needed)
- Doctor Consultation Recommendations (if necessary)
- Rest Time (if relevant)
- Diet Suggestions (if applicable)
- Exercise Recommendations (if relevant)
- Precautions (if necessary)

Provide a comprehensive answer based on the context provided below:

Context: {context}
Question: {question_translated}

Respond in a clear, logical, and user-friendly manner.

---

Title
Provide a clear and concise title summarizing the topic.

---
Understanding Your Symptoms
Briefly explain the symptoms and their potential causes in plain, easy-to-understand language.

---
Immediate Steps 
- List actionable steps the user can take right away to manage their symptoms or situation.

---
Home Remedies
- Include any relevant home remedies that can help alleviate the symptoms or improve the condition.

---
Medication Advice
- Provide recommendations for over-the-counter medications, dosages, and precautions.  
- Emphasize the importance of consulting a doctor if necessary.

---
When to Consult a Doctor 
- Specify conditions or warning signs indicating the need for professional medical attention.

---
Precautions 
- Mention preventive measures or things to avoid that could exacerbate the condition.

---
Diet Suggestions
- Include dietary recommendations to support recovery and manage symptoms.

---
Exercise Recommendations 
- Suggest light exercises or activities, if applicable.  
- Advise against strenuous efforts where necessary.
---
"""

    messages = [{"role": "user", "content": prompt}]

    # Get the response from the Groq model
    groq_model = GroqChat(model_id="llama-3.1-8b-instant")
    response = groq_model(messages)

    # Format the response for display with line breaks
    formatted_response = response.replace("\n", "<br>")

    # Translate the response back to the original language
    response_translated = detect_and_translate(formatted_response, target_lang=user_lang)
    return response_translated

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process_question', methods=["POST"])
def process_question():
    question = request.json.get("question")
    user_lang = request.json.get("language", "en")

    if question:
        response = handle_question(question, vectorstore, user_lang)
        return jsonify({"text": response})
    return jsonify({"error": "No question provided."})

@app.route('/upload_pdf', methods=["POST"])
def upload_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF uploaded."})

    docs = request.files.getlist('pdf')

    if not docs:
        return jsonify({"error": "No files received. Please upload a valid PDF file."})

    try:
        raw_text = get_pdf_text(docs)
        if not raw_text.strip():
            return jsonify({"error": "Uploaded PDFs contain no readable text."})

        text_chunks = get_chunks(raw_text)

        global vectorstore
        vectorstore = get_vectorstore(text_chunks)

        return jsonify({"message": "PDFs processed and ready for questions."})
    except Exception as e:
        return jsonify({"error": f"An error occurred while processing the PDFs: {str(e)}"})

if __name__ == '__main__':
    default_pdf_path = os.getenv("DEFAULT_PDF_PATH")
    if default_pdf_path and os.path.exists(default_pdf_path):
        with open(default_pdf_path, "rb") as f:
            raw_text = get_pdf_text([f])
            text_chunks = get_chunks(raw_text)
            vectorstore = get_vectorstore(text_chunks)
    else:
        vectorstore = None

    app.run(debug=True, host='127.0.0.1', port=int(os.getenv("PORT", 5000)))