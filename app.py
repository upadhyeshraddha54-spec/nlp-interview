
import os, re, json, glob, io
import streamlit.components.v1 as components
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from groq import Groq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Interview Coach", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.hero {background: linear-gradient(120deg, #4f46e5, #06b6d4); padding: 24px 30px; border-radius: 16px; color: white; margin-bottom: 20px;}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: 6px 0 0; opacity: 0.9;}
.chip {display: inline-block; padding: 4px 12px; margin: 3px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;}
.green {background: #dcfce7; color: #166534;}
.red {background: #fee2e2; color: #991b1b;}
.question {background: rgba(99,102,241,0.08); border-left: 4px solid #6366f1; padding: 12px 16px; border-radius: 10px; margin: 8px 0;}
</style>
<div class="hero">
  <h1>🎯 AI Resume Analyzer & Interview Coach</h1>
  <p>Upload your resume, find your skill gaps and practise personalised interview questions</p>
</div>
""", unsafe_allow_html=True)

# ---------------- data ----------------
skill_database = {
    "Programming Languages": ["python", "java", "c++", "c#", "javascript", "sql"],
    "Machine Learning": ["machine learning", "deep learning", "scikit-learn", "tensorflow", "pytorch", "neural networks"],
    "NLP": ["nlp", "natural language processing", "spacy", "nltk", "transformers", "text preprocessing"],
    "Generative AI": ["generative ai", "llm", "large language model", "rag", "retrieval augmented generation",
                      "prompt engineering", "langchain", "langgraph"],
    "Database": ["mysql", "postgresql", "mongodb", "firebase", "sqlite"],
    "Web Development": ["html", "css", "react", "node.js", "fastapi", "flask", "django"],
    "Data Science": ["data analysis", "pandas", "numpy", "matplotlib", "statistics", "data visualization"],
    "Soft Skills": ["communication", "leadership", "teamwork", "problem solving"],
}
job_roles = {
    "AI Engineer": ["python", "machine learning", "deep learning", "generative ai", "llm", "rag", "prompt engineering"],
    "Data Scientist": ["python", "sql", "pandas", "numpy", "machine learning", "statistics", "data analysis"],
    "NLP Engineer": ["python", "nlp", "spacy", "nltk", "transformers", "deep learning"],
    "Backend Developer": ["python", "java", "sql", "mongodb", "fastapi", "flask", "django", "node.js"],
    "Machine Learning Engineer": ["python", "machine learning", "deep learning", "scikit-learn", "tensorflow", "pytorch"],
}
analyzer = SentimentIntensityAnalyzer()


# ---------------- functions ----------------
def extract_text(name, data):
    """name = file name, data = file bytes"""
    if name.lower().endswith(".pdf"):
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    return data.decode("utf-8", errors="ignore")


# Upload box: reads the PDF text in the browser (pdf.js) and sends only the text to the app.
# Small messages pass through Colab tunnels reliably, large file uploads often fail.
UPLOAD_HTML = """<html><body style="margin:0;font-family:sans-serif">
<script src="https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.min.js"></script>
<label style="display:flex;align-items:center;gap:12px;padding:18px;border:2px dashed #6366f1;border-radius:12px;
cursor:pointer;background:rgba(99,102,241,.10);color:#6366f1;font-weight:600;font-size:16px">
<span style="font-size:26px">📄</span><span id="msg">Click here to choose your resume (PDF or TXT)</span>
<input id="f" type="file" accept=".pdf,.txt" style="display:none"></label>
<script>
function send(type, data) { window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, data), "*"); }
send("streamlit:componentReady", {apiVersion: 1});
send("streamlit:setFrameHeight", {height: 80});
var msg = document.getElementById("msg");
async function pdfText(buffer) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@3.11.174/build/pdf.worker.min.js";
  var pdf = await pdfjsLib.getDocument({data: buffer}).promise;
  var text = "";
  for (var i = 1; i <= pdf.numPages; i++) {
    var content = await (await pdf.getPage(i)).getTextContent();
    content.items.forEach(function (it) { text += it.str + (it.hasEOL ? "\\n" : " "); });
    text += "\\n";
  }
  return text;
}
document.getElementById("f").onchange = async function (e) {
  var file = e.target.files[0];
  if (!file) return;
  msg.textContent = "Reading " + file.name + " ...";
  try {
    var text = file.name.toLowerCase().endsWith(".pdf") ? await pdfText(await file.arrayBuffer()) : await file.text();
    send("streamlit:setComponentValue", {value: {name: file.name, text: text}, dataType: "json"});
    msg.textContent = "✅ " + file.name;
  } catch (err) {
    msg.textContent = "❌ Could not read file: " + err;
  }
};
</script></body></html>"""
os.makedirs("resume_upload", exist_ok=True)
with open("resume_upload/index.html", "w") as f:
    f.write(UPLOAD_HTML)
resume_upload = components.declare_component("resume_upload", path="resume_upload")


def clean_text(text):
    text = re.sub(r"[•●▪]", "\n", text)
    text = re.sub(r"[–—]", "-", text)
    text = re.sub(r"(?<=[a-z])(?=(19|20)\d{2}\b)", " ", text)
    text = re.sub(r"(?<=[A-Za-z]):(?=[A-Za-z0-9])", ": ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def extract_skills(text):
    text = text.lower()
    found = {}
    for category, skills in skill_database.items():
        hits = [s for s in skills if re.search(r"(?<!\w)" + re.escape(s) + r"(?![\w+#])", text)]
        if hits:
            found[category] = hits
    return found


def role_scores(text, skills):
    docs = [text.lower()] + [" ".join(v) for v in job_roles.values()]
    matrix = TfidfVectorizer().fit_transform(docs)
    similarity = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    rows = []
    for (role, required), sim in zip(job_roles.items(), similarity):
        match = len([s for s in required if s in skills]) / len(required) * 100
        rows.append({"Job Role": role, "Skill Match %": round(match, 1), "Similarity": round(sim, 3)})
    return pd.DataFrame(rows).sort_values("Skill Match %", ascending=False)


def get_client():
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    typed = st.sidebar.text_input("Groq API Key", type="password",
                                  placeholder="Already set" if key else "Paste your gsk_... key")
    key = typed.strip() or key
    return Groq(api_key=key) if key else None


def ask_llm(client, prompt, json_mode=False):
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_completion_tokens=4000, **extra)
    text = response.choices[0].message.content
    return json.loads(text) if json_mode else text.strip()


def generate_questions(client, skills, role, difficulty, n):
    prompt = f"""You are an expert technical interviewer.
Generate exactly {n} interview questions.
Job Role: {role}
Candidate Skills: {skills}
Difficulty: {difficulty}
Include theoretical, practical and scenario-based questions. No answers. No duplicates.
Return JSON: {{"questions": [{{"question": "...", "skill": "...", "difficulty": "{difficulty}"}}]}}"""
    return ask_llm(client, prompt, json_mode=True).get("questions", [])


def evaluate_answer(client, question, answer, role):
    prompt = f"""You are a technical interviewer for the role {role}.
Question: {question}
Candidate answer: {answer}
Give a score out of 10, strengths, improvements and a short ideal answer.
Return JSON: {{"score": 0, "strengths": [], "improvements": [], "ideal_answer": ""}}"""
    return ask_llm(client, prompt, json_mode=True)


def analyze_sentiment(text):
    text = re.sub(r"(\w)\1{2,}", r"\1\1", text)
    score = analyzer.polarity_scores(text)["compound"]
    return ("Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"), score


def chips(items, color):
    return " ".join(f'<span class="chip {color}">{i}</span>' for i in items) or "None"


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    difficulty = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"], index=1)
    num_questions = st.slider("Number of questions", 3, 15, 5)
    language = st.selectbox("Question language", ["English", "Hindi", "Marathi"])
client = get_client()

# ---------------- upload ----------------
st.markdown("**Upload your resume**")
uploaded = resume_upload(key="resume", default=None)
with st.expander("Other options: use a resume already in Colab or paste the text"):
    local_choice = st.selectbox("Resume found in Colab", ["None"] + sorted(glob.glob("*.pdf")))
    pasted = st.text_area("Or paste your resume text here", height=150)

if uploaded:
    raw_text = uploaded["text"]
elif local_choice != "None":
    raw_text = extract_text(local_choice, open(local_choice, "rb").read())
elif pasted.strip():
    raw_text = pasted
else:
    st.info("Upload a resume to start.")
    st.stop()

resume_text = clean_text(raw_text)
if not resume_text:
    st.error("No text found. The PDF may be scanned.")
    st.stop()

skills_found = extract_skills(resume_text)
all_skills = [s for v in skills_found.values() for s in v]
scores = role_scores(resume_text, all_skills)
role = st.sidebar.selectbox("Job role", list(scores["Job Role"]))
required = job_roles[role]
matched = [s for s in required if s in all_skills]
missing = [s for s in required if s not in all_skills]

tab1, tab2, tab3 = st.tabs(["📊 Resume Analysis", "🎤 Interview Practice", "💬 Feedback"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Skills found", len(all_skills))
    c2.metric("Best matching role", scores.iloc[0]["Job Role"])
    c3.metric(f"Match for {role}", f"{round(len(matched) / len(required) * 100)}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Extracted details")
        st.write("**Email:**", re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", resume_text))
        st.write("**Phone:**", re.findall(r"\+?\d{2}[\s-]?\d{5}[\s-]?\d{5}|\+?\d{2}[\s-]?\d{10}|\b\d{10}\b", resume_text))
        st.subheader("Skills by category")
        for category, skills in skills_found.items():
            st.markdown(f"**{category}**<br>{chips(skills, 'green')}", unsafe_allow_html=True)
    with right:
        st.subheader(f"Skill gap for {role}")
        st.markdown("✅ Matched<br>" + chips(matched, "green"), unsafe_allow_html=True)
        st.markdown("❌ Missing<br>" + chips(missing, "red"), unsafe_allow_html=True)
        st.subheader("Role matching")
        st.bar_chart(scores.set_index("Job Role")["Skill Match %"])
    with st.expander("View extracted resume text"):
        st.text(resume_text)

with tab2:
    if st.button("🚀 Generate Questions", type="primary"):
        if client is None:
            st.error("Groq API key not found. Paste it in the sidebar (Groq API Key box).")
        else:
            try:
                with st.spinner("Generating questions..."):
                    questions = generate_questions(client, matched or all_skills[:6], role, difficulty, num_questions)
                    if language != "English":
                        text = "\n".join(q["question"] for q in questions)
                        translated = ask_llm(client, f"Translate each line into {language}. Keep technical words in English. "
                                                     f"Return only the translated lines.\n\n{text}")
                        for q, t in zip(questions, [l for l in translated.split("\n") if l.strip()]):
                            q["translated"] = t
                    st.session_state["questions"] = questions
            except Exception as e:
                st.error(f"Could not generate questions: {e}")

    for i, q in enumerate(st.session_state.get("questions", [])):
        st.markdown(f'<div class="question"><b>Q{i + 1}.</b> {q.get("translated", q["question"])}'
                    f'<br><small>Skill: {q.get("skill", "")}</small></div>', unsafe_allow_html=True)
        with st.expander(f"Answer Q{i + 1}"):
            answer = st.text_area("Your answer", key=f"answer_{i}")
            if st.button("Evaluate", key=f"eval_{i}") and answer.strip():
                if client is None:
                    st.error("Add your Groq API key in the sidebar.")
                    st.stop()
                with st.spinner("Evaluating..."):
                    result = evaluate_answer(client, q["question"], answer, role)
                st.metric("Score", f"{result.get('score')}/10")
                st.write("**Strengths:**", result.get("strengths"))
                st.write("**Improvements:**", result.get("improvements"))
                st.info(result.get("ideal_answer"))

with tab3:
    feedback = st.text_area("Your feedback about the questions (any language)")
    if st.button("Analyse Sentiment") and feedback.strip():
        english = feedback
        if client is not None and not feedback.isascii():
            english = ask_llm(client, f"Translate into English. Return only the translation.\n\n{feedback}")
        label, score = analyze_sentiment(english)
        if english != feedback:
            st.write("**English:**", english)
        c1, c2 = st.columns(2)
        c1.metric("Sentiment", label)
        c2.metric("Compound score", score)
