import streamlit as st
import joblib
import PyPDF2
import re
import string
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd

# ------------------------------
# Load pre-trained model and vectorizer
# ------------------------------
lr_model = joblib.load("resume_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ------------------------------
# Function to clean resume text
# ------------------------------
def cleanResume(txt):
    cleanText = txt.lower()
    cleanText = re.sub(r'http\S+|www\S+', ' ', cleanText)       
    cleanText = re.sub(r'\S+@\S+', ' ', cleanText)              
    cleanText = re.sub(r'\b\d{10}\b', ' ', cleanText)           
    cleanText = re.sub(r'\+?\d[\d -]{8,}\d', ' ', cleanText)    
    cleanText = re.sub(r'<.*?>', ' ', cleanText)                
    cleanText = re.sub(r'\brt\b|\bcc\b', ' ', cleanText)        
    cleanText = re.sub(r'#\S+|@\S+', ' ', cleanText)            
    cleanText = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', cleanText)  
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)         
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()          
    return cleanText

# ------------------------------
# Extract text functions
# ------------------------------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    uploaded_file.seek(0)  
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload PDF or TXT.")
    return text

# ------------------------------
# Prediction function
# ------------------------------
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    predicted_category = lr_model.predict(vectorized_text)
    return predicted_category[0] 

# ------------------------------
# Skill Extraction
# ------------------------------
skills_df = pd.read_csv("skills.csv")  
# print(skills_df)
skills_list = skills_df["skill"].dropna().unique().tolist()
# print(skills_list)

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill) for skill in skills_list]
matcher.add("SKILLS", patterns)

def extract_skills(resume_text):
    doc = nlp(resume_text.lower())
    matches = matcher(doc)
    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text)
    return list(found_skills)

# ------------------------------
# Streamlit App
# ------------------------------
def main():
    st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")
    st.title("Resume Analyzer App")
    st.markdown("Upload a resume in PDF or TXT format to get the predicted job category, extracted skills, and relevant jobs.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Text extracted successfully from the resume!")

            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            # Predicted Category
            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")

            # Extracted Skills
            st.subheader("Extracted Skills")
            skills = extract_skills(resume_text)
            if skills:
                st.write(", ".join(skills))
            else:
                st.write("No skills found in the resume.")


        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
