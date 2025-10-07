import streamlit as st
import joblib
import PyPDF2
import re
import string
import spacy
from spacy.matcher import PhraseMatcher
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter

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
skills_list = skills_df["skill"].dropna().unique().tolist()

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
    st.title("📄 Resume Analyzer App")

    # Sidebar for user type selection
    user_type = st.sidebar.radio("Select User Type", ["Candidate", "HR"])
    st.sidebar.markdown("---")
    st.sidebar.info("Built with ❤️ using Streamlit")

    # ================================
    # CANDIDATE SECTION
    # ================================
    if user_type == "Candidate":
        st.markdown("👤 **Candidate Mode:** Upload a resume to get job category prediction and skill extraction.")

        uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "txt"])

        if uploaded_file is not None:
            try:
                resume_text = handle_file_upload(uploaded_file)
                st.success("✅ Text extracted successfully from the resume!")

                if st.checkbox("Show extracted text", False):
                    st.text_area("Extracted Resume Text", resume_text, height=300)

                # Predicted Category
                st.subheader("🔍 Predicted Job Category")
                category = pred(resume_text)
                st.write(f"The predicted category of the uploaded resume is: **{category}**")

                # Extracted Skills
                st.subheader("🛠 Extracted Skills")
                skills = extract_skills(resume_text)
                if skills:
                    st.write(", ".join(skills))
                else:
                    st.write("No skills found in the resume.")

            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")

    # ================================
    # HR SECTION
    # ================================
    elif user_type == "HR":
        st.markdown("💼 **HR Mode:** Upload multiple resumes and a job description to rank the top candidates.")

        uploaded_files = st.file_uploader("📂 Upload Multiple Resumes", type=["pdf", "txt"], accept_multiple_files=True)
        job_description = st.text_area("📋 Paste the Job Description", height=200)

        if uploaded_files and job_description:
            if st.button("🚀 Rank Resumes by Relevance"):
                jd_cleaned = cleanResume(job_description)
                jd_vector = tfidf.transform([jd_cleaned])

                ranking_results = []
                for uploaded_file in uploaded_files:
                    try:
                        resume_text = handle_file_upload(uploaded_file)
                        cleaned_resume = cleanResume(resume_text)
                        resume_vector = tfidf.transform([cleaned_resume])
                        similarity = cosine_similarity(resume_vector, jd_vector)[0][0]

                        skills = extract_skills(resume_text)

                        ranking_results.append({
                            "Filename": uploaded_file.name,
                            "Similarity Score (%)": round(similarity * 100, 2),
                            "Skills": skills
                        })

                    except Exception as e:
                        st.warning(f"❌ Could not process {uploaded_file.name}: {str(e)}")

                if ranking_results:
                    sorted_results = sorted(ranking_results, key=lambda x: x["Similarity Score (%)"], reverse=True)
                    df_results = pd.DataFrame([{
                        "Filename": r["Filename"],
                        "Similarity Score (%)": r["Similarity Score (%)"]
                    } for r in sorted_results])

                    st.subheader("🏆 Top Ranked Resumes")
                    st.dataframe(df_results, use_container_width=True)

                    st.download_button(
                        label="⬇️ Download Ranking as CSV",
                        data=df_results.to_csv(index=False),
                        file_name="ranked_resumes.csv",
                        mime="text/csv"
                    )

                    # Skills Details Section with Pie Charts
                    st.subheader("🧠 Skills Details for Each Candidate")

                    for candidate in sorted_results:
                        with st.expander(f"📄 {candidate['Filename']} — {candidate['Similarity Score (%)']}% match"):
                            skills = candidate['Skills']
                            if skills:
                                st.markdown("**Extracted Skills:**")
                                st.write(", ".join(skills))

                                skill_counts = Counter(skills)
                                labels = list(skill_counts.keys())
                                sizes = list(skill_counts.values())

                                fig, ax = plt.subplots()
                                ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                                ax.axis('equal')
                                st.pyplot(fig)
                            else:
                                st.info("No skills found in this resume.")
                else:
                    st.warning("⚠️ No resumes were successfully processed.")

if __name__ == "__main__":
    main()
