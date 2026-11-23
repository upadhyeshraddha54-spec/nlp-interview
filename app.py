
import os
import re
import json
import io

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="AI-Powered Resume Analyzer & Interview Question Generator",
    page_icon="🎯",
    layout="wide"
)

# -------------------------------------------------
# STYLING
# -------------------------------------------------
st.markdown("""
<style>
    :root { --blue: #123B68; --blue-mid: #245C96; --blue-soft: #D5E8FA; --blue-pale: #E6F2FF; --ink: #183452; }
    .stApp { background: #ffffff; color: var(--ink); }
    [data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #DCE4F2; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #123B68 !important; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: var(--ink) !important; }
    [data-testid="stSelectbox"] label, [data-testid="stMultiSelect"] label, [data-testid="stTextInput"] label, [data-testid="stTextArea"] label, [data-testid="stNumberInput"] label, [data-testid="stSlider"] label { color: #123B68 !important; font-weight: 650 !important; opacity: 1 !important; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stMultiSelect"] [data-baseweb="select"] > div { background: #DCEBFA !important; color: #183452 !important; border: 1px solid #8DB6E3 !important; }
    [data-testid="stSelectbox"] [data-baseweb="select"] *, [data-testid="stMultiSelect"] [data-baseweb="select"] * { color: #183452 !important; fill: #245C96 !important; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div { background: #DCEBFA !important; border: 1.5px solid #8DB6E3 !important; border-radius: 8px !important; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * { color: #183452 !important; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] label { color: #245C96 !important; font-weight: 650 !important; }
    [data-baseweb="input"] > div, [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stNumberInput"] input { background: #ffffff !important; color: #183452 !important; border-color: #8DB6E3 !important; }
    [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], [role="option"] { background: #ffffff !important; color: #183452 !important; }
    [role="option"]:hover { background: #eaf3ff !important; }
    /* Sidebar navigation: rectangular blue highlight for the selected page */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label { border-radius: 6px !important; padding: 8px 10px !important; margin: 3px 0 !important; transition: background 0.2s ease; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) { background: #D1E5FA !important; border: 1px solid #6FA6DB !important; box-shadow: inset 4px 0 0 #245C96 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover { background: #DCEBFA !important; }
    [data-testid="stFileUploaderDropzone"] { background: #EAF4FF !important; border: 1px dashed #245C96 !important; }
    [data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] small { color: #183452 !important; }
    [data-testid="stFileUploaderDropzone"] button { background: #245C96 !important; color: #ffffff !important; border: 1px solid #245C96 !important; }
    .brand { text-align: center; padding: 12px 4px 22px 4px; border-bottom: 1px solid #BBD5F0; margin-bottom: 20px; }
    .brand-icon { color: #245C96; font-size: 38px; line-height: 1.2; }
    .brand-title { color: #123B68; font-size: 20px; line-height: 1.35; font-weight: 750; margin-top: 8px; }
    .page-title { color: #123B68 !important; font-size: 36px; font-weight: 750; margin: 0 0 4px 0; }
    .page-subtitle { color: #355578; font-size: 17px; margin-bottom: 18px; }
    .upload-panel { border: 1.5px dashed #245C96; border-radius: 10px; padding: 18px 20px; background: #EAF4FF; margin: 12px 0 22px 0; }
    .upload-heading { color: #123B68 !important; font-size: 20px; font-weight: 700; }
    .upload-caption { color: #355578 !important; font-size: 14px; }
    .metric-card, .content-card { background: #EAF4FF; border: 1px solid #BBD5F0; border-radius: 12px; padding: 18px 20px; margin-bottom: 12px; min-height: 110px; box-shadow: 0 1px 3px rgba(52, 74, 102, 0.04); }
    .metric-label { color: #355578; font-size: 15px; margin-bottom: 8px; }
    .metric-value { color: #123B68; font-size: 27px; line-height: 1.2; font-weight: 750; }
    .card-heading { color: #123B68; font-size: 19px; font-weight: 700; margin-bottom: 14px; }
    .skill-category { color: #123B68; font-weight: 700; margin-top: 10px; margin-bottom: 2px; }
    .skill-line, .plain-list { color: #344a66; margin-bottom: 7px; line-height: 1.5; }
    .plain-list { line-height: 1.9; margin: 0; padding-left: 22px; }
    .match-list { color: #176b45; } .missing-list { color: #a33333; }
    .role-row { display: grid; grid-template-columns: 190px 1fr 55px; gap: 14px; align-items: center; margin: 13px 0; color: #344a66; font-size: 15px; }
    .bar-track { height: 13px; background: #DCEBFA; border-radius: 20px; overflow: hidden; }
    .bar-fill { height: 100%; background: #245C96; border-radius: 20px; }
    .question-card { border: 1px solid #BBD5F0; border-left: 4px solid #245C96; border-radius: 9px; padding: 15px 17px; margin: 10px 0; background: #EAF4FF; color: #183452; }
    div.stButton > button[kind="primary"] { background: #245C96 !important; border: 1px solid #245C96 !important; color: white !important; }
    div.stButton > button[kind="primary"]:hover { background: #123B68 !important; border-color: #123B68 !important; }
    @media (max-width: 800px) { .page-title { font-size: 28px; } .role-row { grid-template-columns: 115px 1fr 45px; gap: 8px; } }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# SKILL AND ROLE DATA
# -------------------------------------------------
skill_database = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "javascript", "sql"
    ],
    "Machine Learning": [
        "machine learning", "deep learning", "scikit-learn",
        "tensorflow", "pytorch", "neural networks"
    ],
    "NLP": [
        "nlp", "natural language processing", "spacy", "nltk",
        "transformers", "text preprocessing"
    ],
    "Generative AI": [
        "generative ai", "llm", "large language model", "rag",
        "retrieval augmented generation", "prompt engineering",
        "langchain", "langgraph"
    ],
    "Database": [
        "mysql", "postgresql", "mongodb", "firebase", "sqlite"
    ],
    "Web Development": [
        "html", "css", "react", "node.js", "fastapi", "flask", "django"
    ],
    "Data Science": [
        "data analysis", "pandas", "numpy", "matplotlib",
        "statistics", "data visualization"
    ],
    "Soft Skills": [
        "communication", "leadership", "teamwork", "problem solving"
    ],
}

job_roles = {
    "AI Engineer": [
        "python", "machine learning", "deep learning",
        "generative ai", "llm", "rag", "prompt engineering"
    ],
    "Data Scientist": [
        "python", "sql", "pandas", "numpy",
        "machine learning", "statistics", "data analysis"
    ],
    "NLP Engineer": [
        "python", "nlp", "spacy", "nltk", "transformers", "deep learning"
    ],
    "Backend Developer": [
        "python", "java", "sql", "mongodb",
        "fastapi", "flask", "django", "node.js"
    ],
    "Machine Learning Engineer": [
        "python", "machine learning", "deep learning",
        "scikit-learn", "tensorflow", "pytorch"
    ],
}

sentiment_analyzer = SentimentIntensityAnalyzer()


# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def extract_resume_text(uploaded_file):
    """Extract text from an uploaded PDF or TXT resume."""
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        pdf = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

    return file_bytes.decode("utf-8", errors="ignore")


def clean_text(text):
    text = re.sub(r"[•●▪]", "\n", text)
    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"(?<=[a-z])(?=(19|20)\d{2}\b)", " ", text)
    text = re.sub(r"(?<=[A-Za-z]):(?=[A-Za-z0-9])", ": ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def extract_skills(text):
    text_lower = text.lower()
    found = {}

    for category, skills in skill_database.items():
        hits = [
            skill for skill in skills
            if re.search(
                r"(?<!\w)" + re.escape(skill) + r"(?![\w+#])",
                text_lower
            )
        ]
        if hits:
            found[category] = hits

    return found


def role_scores(resume_text, all_skills):
    role_documents = [" ".join(required) for required in job_roles.values()]
    documents = [resume_text.lower()] + role_documents

    try:
        matrix = TfidfVectorizer().fit_transform(documents)
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    except ValueError:
        similarities = [0] * len(job_roles)

    rows = []
    for (role_name, required), similarity in zip(job_roles.items(), similarities):
        matched_count = sum(skill in all_skills for skill in required)
        match_percent = matched_count / len(required) * 100

        rows.append({
            "Job Role": role_name,
            "Skill Match %": round(match_percent),
            "Similarity": round(float(similarity), 3)
        })

    return pd.DataFrame(rows).sort_values(
        "Skill Match %", ascending=False
    ).reset_index(drop=True)


def get_groq_client():
    """Read the API key from environment variables or Streamlit secrets only.

    No API-key input is shown in the application sidebar.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "").strip()
        except Exception:
            api_key = ""

    return Groq(api_key=api_key) if api_key else None


def ask_llm(client, prompt, json_mode=False):
    options = {}
    if json_mode:
        options["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=4000,
        **options
    )

    answer = response.choices[0].message.content
    return json.loads(answer) if json_mode else answer.strip()


def generate_questions(client, skills, role, difficulty, number, language):
    prompt = f"""
You are an experienced technical interviewer.
Generate exactly {number} interview questions for the role: {role}.
Candidate skills: {", ".join(skills) if skills else "Not specified"}.
Difficulty: {difficulty}.
Question language: {language}.
Include a useful mix of conceptual, practical and scenario-based questions.
Do not provide answers. Avoid duplicate questions.
Keep technical terms such as Python, SQL, API and machine learning in English.
Return valid JSON in this format:
{{"questions": [
    {{"question": "Question text", "skill": "Related skill"}}
]}}
"""
    return ask_llm(client, prompt, json_mode=True).get("questions", [])


def analyze_sentiment(text):
    normalized = re.sub(r"(\w)\1{2,}", r"\1\1", text)
    score = sentiment_analyzer.polarity_scores(normalized)["compound"]

    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return label, score


def render_metric(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_skill_list(items, css_class=""):
    if not items:
        st.write("None identified.")
        return

    list_items = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f'<ul class="plain-list {css_class}">{list_items}</ul>',
        unsafe_allow_html=True
    )


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="brand">
            <div class="brand-icon">◎</div>
            <div class="brand-title">
                RESUME INSIGHTS<br>
                & INTERVIEW PREP
            </div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Resume Analysis", "Question Generator", "Feedback"],
        label_visibility="collapsed"
    )

    st.divider()
    st.subheader("Settings")

    selected_role = st.selectbox(
        "Job Role",
        list(job_roles.keys())
    )

    difficulty = st.selectbox(
        "Difficulty Level",
        ["Beginner", "Intermediate", "Advanced"],
        index=1
    )

    language = st.selectbox(
        "Language",
        ["English", "Hindi", "Marathi"]
    )

    client = get_groq_client()


# -------------------------------------------------
# RESUME UPLOAD AND ANALYSIS
# -------------------------------------------------
st.markdown(
    '<div class="page-title">AI-Powered Resume Analyzer & Interview Question Generator</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="page-subtitle">Analyze your skills, explore career matches, and practice smarter for your next interview.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="upload-panel">
    <div class="upload-heading">Upload Your Resume</div>
    <div class="upload-caption">Choose a PDF or TXT file to begin your resume analysis.</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a resume (PDF or TXT)",
    type=["pdf", "txt"],
    help="Your resume is analyzed locally by this app."
)

if uploaded_file is None:
    st.info("Upload your resume to view your analysis and generate interview questions.")
    st.stop()

try:
    raw_text = extract_resume_text(uploaded_file)
    resume_text = clean_text(raw_text)
except Exception as error:
    st.error(f"Could not read this file: {error}")
    st.stop()

if not resume_text:
    st.error(
        "No readable text was found. If this is a scanned PDF, convert it "
        "to a text-based PDF or upload a TXT file."
    )
    st.stop()

skills_found = extract_skills(resume_text)
all_skills = [skill for category_skills in skills_found.values()
              for skill in category_skills]

scores = role_scores(resume_text, all_skills)
best_role = scores.iloc[0]["Job Role"]

required_skills = job_roles[selected_role]
matched_skills = [skill for skill in required_skills if skill in all_skills]
missing_skills = [skill for skill in required_skills if skill not in all_skills]
selected_match = round(len(matched_skills) / len(required_skills) * 100)

email_matches = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", resume_text)
phone_matches = re.findall(
    r"\+?\d{2}[\s-]?\d{5}[\s-]?\d{5}|\+?\d{2}[\s-]?\d{10}|\b\d{10}\b",
    resume_text
)


# -------------------------------------------------
# RESUME ANALYSIS PAGE
# -------------------------------------------------
if page == "Resume Analysis":
    st.markdown("## Resume Analysis")

    metric_columns = st.columns(3)
    with metric_columns[0]:
        render_metric("Skills Found", len(all_skills))
    with metric_columns[1]:
        render_metric("Best Role Match", best_role)
    with metric_columns[2]:
        render_metric("Role Match", f"{selected_match}%")

    st.markdown("")

    skill_columns = st.columns(3)

    with skill_columns[0]:
        with st.container(border=True):
            st.markdown("### Skills Identified")
            if skills_found:
                for category, category_skills in skills_found.items():
                    st.markdown(f"**{category}**")
                    st.write(", ".join(category_skills))
            else:
                st.write("No skills from the current skill list were identified.")

    with skill_columns[1]:
        with st.container(border=True):
            st.markdown("### Matched Skills")
            render_skill_list(matched_skills, "match-list")

    with skill_columns[2]:
        with st.container(border=True):
            st.markdown("### Missing Skills")
            render_skill_list(missing_skills, "missing-list")

    with st.container(border=True):
        st.markdown("### Role Comparison")
        for _, row in scores.iterrows():
            role_name = row["Job Role"]
            percentage = int(row["Skill Match %"])
            st.markdown(
                f"""
                <div class="role-row">
                    <div>{role_name}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width:{percentage}%"></div>
                    </div>
                    <div style="text-align:right;font-weight:600">{percentage}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# -------------------------------------------------
# QUESTION GENERATOR PAGE
# -------------------------------------------------
elif page == "Question Generator":
    st.markdown("## Interview Practice")
    st.caption(
        f"Role: {selected_role}  |  Difficulty: {difficulty}  |  Language: {language}"
    )

    number_of_questions = st.selectbox(
        "Number of questions",
        [3, 5, 7, 10],
        index=1
    )

    if st.button("Generate Interview Questions", type="primary"):
        if client is None:
            st.error("AI question generation is unavailable because GROQ_API_KEY is not configured. Set it as an environment variable or in Streamlit secrets.")
        else:
            try:
                with st.spinner("Preparing interview questions..."):
                    questions = generate_questions(
                        client,
                        matched_skills or all_skills[:8],
                        selected_role,
                        difficulty,
                        number_of_questions,
                        language
                    )
                    st.session_state["generated_questions"] = questions
                    st.session_state["question_role"] = selected_role
            except Exception as error:
                st.error(f"Could not generate questions: {error}")

    questions = st.session_state.get("generated_questions", [])

    if questions:
        st.markdown("### Your Interview Questions")

        for index, question_data in enumerate(questions):
            question_text = question_data.get("question", "")
            related_skill = question_data.get("skill", "General")

            st.markdown(
                f"""
                <div class="question-card">
                    <strong>Question {index + 1}</strong><br>
                    {question_text}
                    <br><small>Related skill: {related_skill}</small>
                </div>
                """,
                unsafe_allow_html=True
            )




# -------------------------------------------------
# FEEDBACK PAGE
# -------------------------------------------------
elif page == "Feedback":
    st.markdown("## Feedback")
    st.write("Share your feedback about the interview questions or your practice experience.")

    feedback = st.text_area(
        "Your feedback",
        placeholder="Write your feedback here...",
        height=160
    )

    if st.button("Analyze Feedback", type="primary"):
        if not feedback.strip():
            st.warning("Please enter feedback first.")
        else:
            feedback_for_analysis = feedback

            if client is not None and not feedback.isascii():
                try:
                    feedback_for_analysis = ask_llm(
                        client,
                        "Translate the following feedback into English. "
                        "Return only the translation:\n\n" + feedback
                    )
                except Exception:
                    feedback_for_analysis = feedback

            sentiment, sentiment_score = analyze_sentiment(feedback_for_analysis)

            feedback_columns = st.columns(2)
            with feedback_columns[0]:
                render_metric("Sentiment", sentiment)
            with feedback_columns[1]:
                render_metric("Sentiment Score", f"{sentiment_score:.2f}")

            if feedback_for_analysis != feedback:
                st.write("**English translation:**", feedback_for_analysis)
