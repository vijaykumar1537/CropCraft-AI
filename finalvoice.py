import os
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
from deep_translator import GoogleTranslator  # Changed from googletrans
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
#from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS


load_dotenv()
os.environ["GOOGLE_API_KEY"] = "AIzaSyCcqE-LgjOYAJ-BSnJIAIlZqPczfpCETLs"  

st.set_page_config(page_title="CropCraft-AI", layout="wide")

engine = pyttsx3.init()

def vector_store_exists():
    return os.path.exists("faiss_index")

def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in 
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer: 
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def get_audio_question():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Say your question:")
        audio = r.listen(source)
    try:
        user_question = r.recognize_google(audio)
        st.write(f"You said: {user_question}")
        return user_question
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand your question.")
        return None
    except sr.RequestError:
        st.error("Could not request results from speech recognition service.")
        return None

def speak_answers(answers, lang='te'):
    try:
        for i, answer in enumerate(answers):
            # Using deep_translator instead of googletrans
            translator = GoogleTranslator(source='en', target=lang)
            translated_answer = translator.translate(answer)
            
            # Create and save the audio file
            tts = gTTS(text=translated_answer, lang=lang)
            filename = f'answer_{i}.mp3'
            tts.save(filename)
            
            # Display the translated text and play the audio
            st.write(f"Translated Answer: {translated_answer}")
            st.audio(filename)
            
            # Clean up the audio file
            try:
                os.remove(filename)
            except:
                pass
    except Exception as e:
        st.error(f"Translation Error: {str(e)}")
        st.write("Displaying original answer instead:")
        for answer in answers:
            st.write(answer)

def main():
    st.header("Crop Craft AI")

    # Language selection with proper codes
    lang_options = {
        'English': 'en',
        'Hindi': 'hi',
        'Telugu': 'te',
        'Tamil': 'ta'
    }
    
    selected_lang = st.selectbox('Select a language:', list(lang_options.keys()))
    lang_code = lang_options[selected_lang]

    if not vector_store_exists():
        with st.spinner("Processing documents..."):
            pdf_dir = "."  
            pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
            if not pdf_files:
                st.warning("No PDF files found in the directory.")
                return
                
            raw_text = ''
            for pdf_file in pdf_files:
                raw_text += get_pdf_text(pdf_file)
            
            if not raw_text.strip():
                st.warning("No text could be extracted from the PDF files.")
                return
                
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Documents processed successfully!")
    
    # Create columns for the buttons
    col1, col2 = st.columns(2)
    
    with col1:
        text_input = st.button("Ask Question with Text")
    with col2:
        voice_input = st.button("Ask Question with Voice")

    user_questions = []
    
    if voice_input:
        user_question = get_audio_question()
        if user_question:
            user_questions.append(user_question)
    else:
        user_question = st.text_input("Please Ask Your Queries Regarding Agriculture and Livestock", key="user_question")
        if user_question:
            user_questions.append(user_question)

    if user_questions:
        with st.spinner("Processing your question..."):
            try:
                answers = []
                for question in user_questions:
                    answer = user_input(question)
                    answers.append(answer)
                    
                if selected_lang != 'English':
                    speak_answers(answers, lang=lang_code)
                else:
                    for answer in answers:
                        st.write(answer)
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                engine.stop()

if __name__ == "__main__":
    main()