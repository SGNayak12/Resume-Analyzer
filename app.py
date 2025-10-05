import streamlit as st
import joblib
# import docx  # Extract text from Word files
import PyPDF2  # Extract text from PDF
import re
import string

# ------------------------------
# Load pre-trained model and vectorizer
# ------------------------------
lr_model = joblib.load("resume_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
naive_byes=joblib.load("Naive-byes_model.pkl")
# If you used LabelEncoder during training, uncomment below
# le = joblib.load("encoder.pkl")

# ------------------------------
# Function to clean resume text
# ------------------------------
def cleanResume(txt):
    cleanText = txt.lower()
    cleanText = re.sub(r'http\S+|www\S+', ' ', cleanText)       # URLs
    cleanText = re.sub(r'\S+@\S+', ' ', cleanText)              # Emails
    cleanText = re.sub(r'\b\d{10}\b', ' ', cleanText)           # 10-digit numbers
    cleanText = re.sub(r'\+?\d[\d -]{8,}\d', ' ', cleanText)    # International numbers
    cleanText = re.sub(r'<.*?>', ' ', cleanText)                # HTML tags
    cleanText = re.sub(r'\brt\b|\bcc\b', ' ', cleanText)        # RT/cc
    cleanText = re.sub(r'#\S+|@\S+', ' ', cleanText)            # hashtags & mentions
    cleanText = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', cleanText)  # punctuation
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)         # non-ASCII
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()          # extra spaces
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

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    uploaded_file.seek(0)  # Reset pointer
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")
    return text

# ------------------------------
# Prediction function
# ------------------------------
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])
    predicted_category = lr_model.predict(vectorized_text)
    
    # If using LabelEncoder:
    # predicted_category_name = le.inverse_transform(predicted_category)
    # return predicted_category_name[0]
    
    return predicted_category[0] 

# ------------------------------
# Streamlit App
# ------------------------------
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format to get the predicted job category.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Text extracted successfully from the resume!")

            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Predicted Category")
            category = pred(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")

if __name__ == "__main__":
    main()
