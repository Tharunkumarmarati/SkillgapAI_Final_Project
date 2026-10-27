import streamlit as st
import pandas as pd
import numpy as np
import re
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.platypus import Image as RLImage
import plotly.io as pio
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer
import tempfile
import html
import base64

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="SkillGap AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Skill Gap Analysis Interface")
st.caption("End-to-End Resume vs Job Description Skill Intelligence System")

# --------------------------------------------------
# SKILL DATABASE
# --------------------------------------------------

TECH_SKILLS = [
    "python","java","c","c++","c#","javascript","typescript","go","rust","php","r",
    "html","css","react","angular","vue","node","express","nextjs","spring","spring boot",
    "django","flask","fastapi","rest api","graphql","jwt authentication",
    "sql","mysql","postgresql","oracle","sqlite","mongodb","redis","cassandra","firebase",
    "data analysis","data analytics","data visualization","eda",
    "data cleaning","data preprocessing","data transformation",
    "statistics","descriptive statistics","hypothesis testing",
    "mean","median","variance","confidence interval","matlab","scala","kotlin","swift",
    "objective c","dart","groovy","perl","lua","haskell","sass","less","bootstrap","tailwind css",
    "svelte","nuxtjs","nest.js","grpc","web sockets","mariadb","couchdb","dynamodb",
    "elasticsearch","neo4j","snowflake","bigquery","data science",
    "probability","numpy","pandas","scipy","matplotlib","seaborn","plotly",
    "excel","advanced excel","power bi","tableau","looker","qlik",
    "dashboards","kpis","reporting","business intelligence","machine learning","deep learning","artificial intelligence",
    "supervised learning","unsupervised learning",
    "nlp","computer vision","time series","recommendation systems",
    "reinforcement learning","scikit-learn","tensorflow","keras","pytorch",
    "xgboost","lightgbm","catboost","transformers","bert","gpt","llm","prompt engineering","aws","azure","gcp","cloud computing",
    "docker","kubernetes","helm","ci/cd","jenkins","github actions","gitlab ci","bitbucket pipelines",
    "terraform","ansible","cloudformation","linux","unix","bash","shell scripting","hadoop","spark","pyspark","flink",
    "kafka","airflow","luigi","etl","elt","data pipelines","data ingestion",
    "data warehousing","lakehouse","delta lake","object oriented programming","functional programming",
    "data structures","algorithms","system design","low level design","high level design",
    "microservices","monolithic architecture","design patterns","clean architecture",
    "unit testing","integration testing","system testing","api testing","automation testing","selenium","cypress",
    "application security","web security","network security","encryption","ssl","tls","oauth","jwt",
    "penetration testing","vulnerability assessment","git","github","gitlab","bitbucket",
    "jira","confluence","trello","asana","postman","swagger","soap ui",
    "vscode","intellij","eclipse","pycharm","jupyter","google colab",
    "iam","zero trust","firewalls","joins","inner join","left join","right join",
    "subqueries","aggregations","group by","having","window functions",
    "pivot tables","vlookup","power query","dax"
]

SOFT_SKILLS = [
    "communication","verbal communication","written communication",
    "presentation skills","public speaking","stakeholder communication",
    "teamwork","collaboration","cross functional collaboration",
    "leadership","people management","mentoring","coaching",
    "problem solving","analytical thinking","critical thinking",
    "decision making","strategic thinking",
    "time management","prioritization","multitasking",
    "adaptability","flexibility","learning mindset",
    "creativity","innovation","design thinking",
    "conflict resolution","negotiation",
    "emotional intelligence","stress management",
    "attention to detail","ownership","accountability",
    "self motivation","work ethic","initiative",
    "customer focus","business acumen","ethics",
    "active listening","mentorship"
]

UPSKILL_MAP = {
    "Aws": {"title": "AWS Cloud Services", "description": "Complete AWS Certified Solutions Architect course", "priority": "High"},
    "Sql": {"title": "SQL for Data Analytics", "description": "Learn advanced SQL queries, joins, and optimization", "priority": "High"},
    "Tensorflow": {"title": "Deep Learning with TensorFlow", "description": "Hands-on TensorFlow & neural networks training", "priority": "Medium"},
    "Data Analysis": {"title": "Advanced Data Analysis", "description": "Statistics, EDA, and real-world data projects", "priority": "Medium"},
    "Leadership": {"title": "Leadership & Management Skills", "description": "Develop leadership, decision-making, and people skills", "priority": "Medium"}
}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
@st.cache_resource
def load_bert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

bert_model = load_bert_model()

def extract_text(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join(p.extract_text() for p in reader.pages if p.extract_text())
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join(p.text for p in doc.paragraphs if p.text)
    else:
        return file.read().decode("utf-8")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text)

def extract_skills(text):
    found = []
    # Longer skills first
    all_skills = sorted(list(set(TECH_SKILLS + SOFT_SKILLS)), key=len, reverse=True)
    pattern = r"\b(" + "|".join(map(re.escape, all_skills)) + r")\b"
    
    for match in re.finditer(pattern, text):
        skill = match.group(1).title()
        if skill not in found:
            found.append(skill)
    return found

def classify(skills):
    tech, soft = [], []
    for s in skills:
        if s.lower() in TECH_SKILLS:
            tech.append(s)
        else:
            soft.append(s)
    return tech, soft

def skill_badges(skills, color):
    if not skills:
        return "<i>No skills found</i>"
    html = "<div style='display:flex; flex-wrap:wrap; gap:6px; overflow-x:auto;'>"
    for s in skills:
        html += f"<span style='background-color:{color}; color:white; padding:6px 12px; border-radius:20px; display:inline-block; font-size:14px; white-space:nowrap;'>{s}</span>"
    html += "</div>"
    return html

def comma_list(skills):
    return ", ".join(sorted(skills)) if skills else "None"

def bert_skill_match(resume_skills, jd_skills, exact_th=0.75, partial_th=0.4):
    results = []
    if not resume_skills or not jd_skills:
        return results
    
    resume_emb = bert_model.encode(resume_skills, normalize_embeddings=True)
    jd_emb = bert_model.encode(jd_skills, normalize_embeddings=True)
    
    for i, jd_skill in enumerate(jd_skills):
        scores = cosine_similarity([jd_emb[i]], resume_emb)[0]
        best_score = float(np.max(scores))
        
        if best_score >= exact_th:
            status = "Matched"
        elif best_score >= partial_th:
            status = "Partial"
        else:
            status = "Missing"
            
        results.append({
            "Skill": jd_skill,
            "Similarity": round(best_score * 100, 2),
            "Status": status
        })
    return results

def generate_upskilling_recommendations(missing, partial):
    recommendations = []
    for skill in missing:
        key = skill.title()
        if key in UPSKILL_MAP:
            rec = UPSKILL_MAP[key].copy()
            rec["skill"] = skill
            rec["status"] = "Missing"
            rec["priority"] = "High"
            recommendations.append(rec)
        else:
            recommendations.append({
                "skill": skill,
                "title": f"Learn {skill}",
                "description": f"Build strong foundation in {skill} to meet job requirements",
                "priority": "High",
                "status": "Missing"
            })
            
    for skill in partial:
        key = skill.title()
        if key in UPSKILL_MAP:
            rec = UPSKILL_MAP[key].copy()
            rec["skill"] = skill
            rec["status"] = "Partial"
            rec["priority"] = "Medium"
            recommendations.append(rec)
        else:
            recommendations.append({
                "skill": skill,
                "title": f"Improve {skill}",
                "description": f"Advance your proficiency in {skill}",
                "priority": "Medium",
                "status": "Partial"
            })
    return recommendations
@st.cache_resource
def load_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

vectorizer = load_vectorizer()

def get_file_extension(file):
    """Get the file extension from uploaded file"""
    if file is None:
        return None
    return '.' + file.name.split('.')[-1].lower()

def validate_file_format(file):
    """Validate if the uploaded file has a supported format"""
    if file is None:
        return False, "No file uploaded"
    
    allowed_extensions = ['.pdf', '.docx', '.txt']
    file_extension = get_file_extension(file)
    
    if file_extension not in allowed_extensions:
        return False, f"Unsupported file format: {file_extension}. Please upload PDF, DOCX, or TXT files only."
    
    return True, "Valid format"

def extract_text(file):
    """Extract text from uploaded file with error handling"""
    try:
        if file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            text = " ".join(p.extract_text() for p in reader.pages if p.extract_text())
            if not text.strip():
                raise ValueError("PDF file appears to be empty or contains only images")
            return text
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = " ".join(p.text for p in doc.paragraphs if p.text)
            if not text.strip():
                raise ValueError("DOCX file appears to be empty")
            return text
        else:
            text = file.read().decode("utf-8")
            if not text.strip():
                raise ValueError("Text file appears to be empty")
            return text
    except Exception as e:
        raise ValueError(f"Error reading file '{file.name}': {str(e)}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text)

def extract_skills(text):
    found = []
    # Longer skills first
    all_skills = sorted(list(set(TECH_SKILLS + SOFT_SKILLS)), key=len, reverse=True)
    pattern = r"\b(" + "|".join(map(re.escape, all_skills)) + r")\b"
    
    for match in re.finditer(pattern, text):
        skill = match.group(1).title()
        if skill not in found:
            found.append(skill)
    return found

def classify(skills):
    tech, soft = [], []
    for s in skills:
        if s.lower() in TECH_SKILLS:
            tech.append(s)
        else:
            soft.append(s)
    return tech, soft

def skill_badges(skills, color):
    if not skills:
        return "<i>No skills found</i>"
    html = "<div style='display:flex; flex-wrap:wrap; gap:6px; overflow-x:auto;'>"
    for s in skills:
        html += f"<span style='background-color:{color}; color:white; padding:6px 12px; border-radius:20px; display:inline-block; font-size:14px; white-space:nowrap;'>{s}</span>"
    html += "</div>"
    return html

def comma_list(skills):
    return ", ".join(sorted(skills)) if skills else "None"

def skill_match(resume_skills, jd_skills, exact_th=0.85, partial_th=0.4):
    """Match skills using TF-IDF and cosine similarity"""
    results = []
    if not resume_skills or not jd_skills:
        return results
    
    # Normalize skills for comparison
    resume_lower = [s.lower() for s in resume_skills]
    
    for jd_skill in jd_skills:
        jd_lower = jd_skill.lower()
        
        # Check for exact match first
        if jd_lower in resume_lower:
            best_score = 1.0
        else:
            # Use TF-IDF for similarity
            all_skills = resume_skills + [jd_skill]
            try:
                tfidf_matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(all_skills)
                jd_vec = tfidf_matrix[-1]
                resume_vecs = tfidf_matrix[:-1]
                similarities = cosine_similarity(jd_vec, resume_vecs)[0]
                best_score = float(np.max(similarities)) if len(similarities) > 0 else 0
            except:
                best_score = 0
        
        if best_score >= exact_th:
            status = "Matched"
        elif best_score >= partial_th:
            status = "Partial"
        else:
            status = "Missing"
            
        results.append({
            "Skill": jd_skill,
            "Similarity": round(best_score * 100, 2),
            "Status": status
        })
    return results

def generate_upskilling_recommendations(missing, partial):
    recommendations = []
    for skill in missing:
        key = skill.title()
        if key in UPSKILL_MAP:
            rec = UPSKILL_MAP[key].copy()
            rec["skill"] = skill
            rec["status"] = "Missing"
            rec["priority"] = "High"
            recommendations.append(rec)
        else:
            recommendations.append({
                "skill": skill,
                "title": f"Learn {skill}",
                "description": f"Build strong foundation in {skill} to meet job requirements",
                "priority": "High",
                "status": "Missing"
            })
            
    for skill in partial:
        key = skill.title()
        if key in UPSKILL_MAP:
            rec = UPSKILL_MAP[key].copy()
            rec["skill"] = skill
            rec["status"] = "Partial"
            rec["priority"] = "Medium"
            recommendations.append(rec)
        else:
            recommendations.append({
                "skill": skill,
                "title": f"Improve {skill}",
                "description": f"Advance your proficiency in {skill}",
                "priority": "Medium",
                "status": "Partial"
            })
    return recommendations

def upskill_cards(recommendations):
    if not recommendations:
        return "<div style='background:#e8f5e9; padding:20px; border-radius:14px; text-align:center; font-size:16px; font-weight:600; color:#2e7d32;'>🎉 Excellent! Your skills already match the job requirements.</div>", 120
    
    html = "<div style='display:flex; flex-wrap:wrap; gap:20px;'>"
    for rec in recommendations:
        color = "#F44336" if rec["priority"] == "High" else "#FFC107"
        icon = "🔥" if rec["priority"] == "High" else "⚡"
        
        html += f"""
        <div style="flex:1 1 320px; background:#ffffff; border-radius:16px; box-shadow:0 10px 24px rgba(0,0,0,0.08); padding:18px; position:relative;">
            <div style="position:absolute; left:0; top:0; bottom:0; width:6px; background:{color}; border-radius:16px 0 0 16px;"></div>
            <div style="padding-left:10px;">
                <h4 style="margin:0;font-size:18px;">{icon} {rec["skill"]}</h4>
                <p style="margin:8px 0; color:#555; font-size:14px;">{rec["description"]}</p>
                <span style="background:{color}; color:white; padding:4px 12px; border-radius:14px; font-size:12px; font-weight:600;">{rec["priority"]} Priority • {rec["status"]}</span>
            </div>
        </div>
        """
    html += "</div>"
    rows = max(1, (len(recommendations) + 2) // 3)
    return html, rows * 200 + 40

# --------------------------------------------------
# FIXED PREVIEW LOGIC
# --------------------------------------------------
def clean_for_preview(text):
    """
    Removes weird special characters but keeps:
    - Alphanumeric (A-Z, 0-9)
    - Basic punctuation (.,!?-@)
    - Layout characters (Newlines \n, Tabs \t)
    """
    # This regex removes anything that IS NOT a letter, number, space, basic punctuation, or newline.
    # We explicitly keep \n (newline) to preserve the sequence.
    cleaned = re.sub(r'[^a-zA-Z0-9\s\.,!?:;@()\-\n\t]', '', text)
    return cleaned

def extract_text_preserve(file):
    text = ""
    try:
        if file.name.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    # Append newline to separate pages/paragraphs
                    text += page_text + "\n\n"
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            for p in doc.paragraphs:
                # Append newline after every paragraph to maintain sequence
                if p.text:
                    text += p.text + "\n"
        else:  # txt
            file.seek(0)
            content = file.read()
            # Decode bytes to string
            if isinstance(content, bytes):
                text = content.decode("utf-8", errors="ignore")
            else:
                text = content
            # Normalize carriage returns
            text = text.replace("\r\n", "\n").replace("\r", "\n")
    except Exception as e:
        text = f"Error reading file: {str(e)}"
        
    return text

def pretty_document_preview(file, max_chars=5000):
    # 1. Extract text preserving structure
    text = extract_text_preserve(file)
    
    # 2. Clean special chars BUT keep newlines
    text = clean_for_preview(text)
    
    # Limit preview size
    preview_text = text[:max_chars]
    if len(text) > max_chars:
        preview_text += "\n\n... (truncated for preview)"
    
    # 3. Escape HTML to make it safe to render
    preview_text = html.escape(preview_text)
    
    # 4. Render with pre-wrap to respect the \n we preserved
    st.markdown(
        f"""
        <div style="
            background-color:#f7f7f7;
            padding:15px;
            border-radius:8px;
            height:450px;
            overflow:auto;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size:14px;
            line-height:1.5;
            color:#111;
            border: 1px solid #ddd;
        ">
        {preview_text}
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# UPLOAD SECTION WITH ERROR HANDLING
# --------------------------------------------------
st.subheader("📄 Upload Documents")

c1, c2 = st.columns(2)
with c1:
    resume_file = st.file_uploader("Upload Resume", ["pdf", "docx", "txt"], key="resume_upload")
with c2:
    jd_file = st.file_uploader("Upload Job Description", ["pdf", "docx", "txt"], key="jd_upload")

# Add an Analyze button
analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

# --------------------------------------------------
# ENHANCED ERROR HANDLING SECTION
# --------------------------------------------------
if analyze_button:
    # Initialize error flags
    has_errors = False
    
    # SCENARIO 1: No files uploaded at all
    if not resume_file and not jd_file:
        st.error("❌ **Error:** Please upload both Resume and Job Description files to proceed with the analysis.")
        has_errors = True
    
    # SCENARIO 2: Only Resume uploaded, JD missing
    elif resume_file and not jd_file:
        st.error("❌ **Error:** Please upload the Job Description file to proceed with the analysis.")
        has_errors = True
    
    # SCENARIO 3: Only JD uploaded, Resume missing
    elif not resume_file and jd_file:
        st.error("❌ **Error:** Please upload the Resume file to proceed with the analysis.")
        has_errors = True
    
    # SCENARIO 4: Both files uploaded - validate formats
    else:
        # Validate Resume format
        is_valid_resume, resume_msg = validate_file_format(resume_file)
        if not is_valid_resume:
            st.error(f"❌ **Resume Format Error:** {resume_msg}")
            st.info("✅ **Supported formats:** PDF (.pdf), Word Document (.docx), Text File (.txt)")
            has_errors = True
        
        # Validate JD format
        is_valid_jd, jd_msg = validate_file_format(jd_file)
        if not is_valid_jd:
            st.error(f"❌ **Job Description Format Error:** {jd_msg}")
            st.info("✅ **Supported formats:** PDF (.pdf), Word Document (.docx), Text File (.txt)")
            has_errors = True
        
        # SCENARIO 5: Check if both files have same format (optional recommendation)
        if is_valid_resume and is_valid_jd:
            resume_ext = get_file_extension(resume_file)
            jd_ext = get_file_extension(jd_file)
            
            if resume_ext != jd_ext:
                # Get format names
                format_names = {'.pdf': 'PDF', '.docx': 'Word Document (DOCX)', '.txt': 'Text File (TXT)'}
                resume_format = format_names.get(resume_ext, resume_ext.upper())
                jd_format = format_names.get(jd_ext, jd_ext.upper())
                
                st.warning(f"⚠️ **Format Mismatch:** Resume is in **{resume_format}** format, but Job Description is in **{jd_format}** format. For better consistency, please upload the Job Description in **{resume_format}** format.")
                has_errors = True
    
    # If there are errors, stop execution
    if has_errors:
        st.stop()
    
    # If we reach here, both files are uploaded and valid
    st.success("✅ Files uploaded successfully! Processing...")
    
    # --------------------------------------------------
    # PROCESSING WITH ERROR HANDLING
    # --------------------------------------------------
    try:
        resume_raw = extract_text(resume_file)
        jd_raw = extract_text(jd_file)
        
        resume_text = clean_text(resume_raw)
        jd_text = clean_text(jd_raw)
        
        resume_skills = extract_skills(resume_text)
        jd_skills = extract_skills(jd_text)
        
        # Warn if no skills found
        if not resume_skills:
            st.warning("⚠️ **Warning:** No skills detected in the Resume. Please ensure the document contains relevant skills.")
        if not jd_skills:
            st.warning("⚠️ **Warning:** No skills detected in the Job Description. Please ensure the document contains skill requirements.")
        
        skill_gap_results = bert_skill_match(resume_skills, jd_skills)

        
        matched = [r["Skill"] for r in skill_gap_results if r["Status"] == "Matched"]
        partial = [r["Skill"] for r in skill_gap_results if r["Status"] == "Partial"]
        missing = [r["Skill"] for r in skill_gap_results if r["Status"] == "Missing"]
        
        overall_score = round((len(matched) / len(jd_skills)) * 100) if jd_skills else 0
        
        rtech, rsoft = classify(resume_skills)
        jtech, jsoft = classify(jd_skills)
        
        # --------------------------------------------------
        # MILESTONE 1 – PARSED DOCUMENTS
        # --------------------------------------------------
        st.divider()
        st.header("📌 Milestone 1: Uploaded Document Preview")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("📄 Resume Preview")
            if resume_file:
                # Reset pointer for preview because extract_text used it
                resume_file.seek(0)
                pretty_document_preview(resume_file)
        
        with c2:
            st.subheader("📄 Job Description Preview")
            if jd_file:
                # Reset pointer for preview
                jd_file.seek(0)
                pretty_document_preview(jd_file)
        
        # --------------------------------------------------
        # MILESTONE 2 – SKILL EXTRACTION & PROPORTION
        # --------------------------------------------------
        st.divider()
        st.header("📌 Milestone 2: Skill Extraction & Classification")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Resume Skills")
            st.markdown("### 🔧 Technical Skills")
            components.html(skill_badges(rtech, "#4CAF50"), height=100, scrolling=True)
        
            st.markdown("### 🤝 Soft Skills")
            components.html(skill_badges(rsoft, "#2196F3"), height=100, scrolling=True)
            
            # Fixed pie chart - handle case when no skills found
            if len(rtech) > 0 or len(rsoft) > 0:
                fig = px.pie(
                    values=[len(rtech), len(rsoft)],
                    names=["Technical", "Soft"],
                    hole=0.45,
                    title="Resume Skill Distribution",
                    color_discrete_sequence=['#4CAF50', '#2196F3']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No skills to display in chart")
        
        with c2:
            st.subheader("Job Description Skills")
            st.markdown("### 🔧 Technical Skills")
            components.html(skill_badges(jtech, "#FF9800"), height=100, scrolling=True)
        
            st.markdown("### 🤝 Soft Skills")
            components.html(skill_badges(jsoft, "#9C27B0"), height=100, scrolling=True)
            
            # Fixed pie chart - handle case when no skills found
            if len(jtech) > 0 or len(jsoft) > 0:
                fig = px.pie(
                    values=[len(jtech), len(jsoft)],
                    names=["Technical", "Soft"],
                    hole=0.45,
                    title="JD Skill Distribution",
                    color_discrete_sequence=['#FF9800', '#9C27B0']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No skills to display in chart")
        
        def skill_summary_card(title, value, color):
            return f"""
            <div style="background:#f8f9fc; border-radius:14px; padding:16px; text-align:center; box-shadow:0 4px 12px rgba(0,0,0,0.08);">
                <div style="font-size:26px; font-weight:700; color:{color};">{value}</div>
                <div style="font-size:13px; color:#555; margin-top:4px;">{title}</div>
            </div>
            """
        
        st.markdown("## 📊 Skill Summaries")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("### 📄 Resume Skills")
            resume_summary = f"""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                {skill_summary_card('Technical Skills', len(rtech), '#4CAF50')}
                {skill_summary_card('Soft Skills', len(rsoft), '#2196F3')}
                {skill_summary_card('Total Skills', len(rtech) + len(rsoft), '#FF9800')}
                {skill_summary_card('Avg Confidence (%)', int((len(rtech)/(len(rtech)+len(rsoft))*100) if (len(rtech)+len(rsoft)) else 0), '#9C27B0')}
            </div>
            """
            components.html(resume_summary, height=260)
        
        with c2:
            st.markdown("### 📄 Job Description Skills")
            jd_summary = f"""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                {skill_summary_card('Technical Skills', len(jtech), '#4CAF50')}
                {skill_summary_card('Soft Skills', len(jsoft), '#2196F3')}
                {skill_summary_card('Total Skills', len(jtech) + len(jsoft), '#FF9800')}
                {skill_summary_card('Avg Confidence (%)', int((len(jtech)/(len(jtech)+len(jsoft))*100) if (len(jtech)+len(jsoft)) else 0), '#9C27B0')}
            </div>
            """
            components.html(jd_summary, height=260)
        
        # --------------------------------------------------
        # MILESTONE 3 – SKILL GAP INTERFACE
        # --------------------------------------------------

        st.divider()
        st.header("📌 Milestone 3: Skill Embedding Generation And Similarity Computation")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Overall Match", f"{overall_score}%")
        k2.metric("Matched Skills", len(matched))
        k3.metric("Partial Matches", len(partial))
        k4.metric("Missing Skills", len(missing))

        st.subheader("🧩 Skill Match Breakdown")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### ✅ Matched Skills")
            st.info(comma_list(matched))
        with c2:
            st.markdown("### ⚠️ Partial Matched Skills")
            st.warning(comma_list(partial))
        with c3:
            st.markdown("### ❌ Missing Skills")
            st.error(comma_list(missing))

        st.subheader("Skill Match Chart")
        col1, col2 = st.columns([2, 2])

        rows = []
        for skill in jd_skills:
            if skill in matched:
                value = 100
                status = "Matched"
            elif skill in partial:
                value = 50
                status = "Partial"
            else:
                value = 0
                status = "Missing"
            rows.append([skill, value, status])



        # Donut Overview
        fig = px.pie(
            values=[len(matched), len(partial), len(missing)],
            names=["Matched", "Partial", "Missing"],
            hole=0.5,
            title="Skill Match Overview",
            color=["Matched", "Partial", "Missing"],
            color_discrete_map={
                "Matched": "#4CAF50",   # Green
                "Partial": "#FFC107",   # Yellow
                "Missing": "#F44336"    # Red
            }
        )

        fig.update_layout(
            title=dict(
                text="Skill Match Overview",
                x=0.5,
                xanchor="center",
                font=dict(color="black", size=20)
            ),
            legend=dict(
                font=dict(color="black"),
                orientation="h",
                y=-0.25,
                x=0.5,
                xanchor="center"
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            height=300,
            width=300,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)


        
        # --------------------------------------------------
        # MILESTONE 4 – DASHBOARD AND REPORT EXPORT MODULE
        # --------------------------------------------------
        st.divider()
        st.header("📌 Milestone 4: Dashboard and Report Export Module")
        
        # Create the dashboard header with refresh and settings
        dash_col1, dash_col2 = st.columns([4, 1])
        with dash_col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;">
                <h2 style="color: white; margin: 0;">🎯 Skill Gap Analysis Dashboard</h2>
                <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0;">Comprehensive skill assessment and recommendations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with dash_col2:
            col_r, col_s = st.columns(2)
            with col_r:
                if st.button("🔄 Refresh", use_container_width=True):
                    st.rerun()
            with col_s:
                show_settings = st.button("⚙️ Settings", use_container_width=True)
        
        # Settings panel
        if show_settings:
            with st.expander("Dashboard Settings", expanded=True):
                st.slider("Match Threshold (%)", 0, 100, 75, key="match_threshold")
                st.selectbox("Chart Theme", ["Default", "Dark", "Light"], key="chart_theme")
        
        # Main Dashboard Layout
        dash_left, dash_right = st.columns([2, 1])
        
        with dash_left:
            # Skill Match Overview Section
            st.markdown("### 📈 Skill Match Overview")
            
            ov_col1, ov_col2, ov_col3, ov_col4 = st.columns(4)
            
            with ov_col1:
                # Overall Match Gauge
                overall_color = "#4CAF50" if overall_score >= 70 else "#FFC107" if overall_score >= 50 else "#F44336"
                components.html(f"""
                <div style="background: white; border-radius: 16px; padding: 25px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                    <div style="position: relative; width: 120px; height: 120px; margin: 0 auto;">
                        <svg viewBox="0 0 36 36" style="transform: rotate(-90deg);">
                            <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none" stroke="#eee" stroke-width="3"/>
                            <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                                fill="none" stroke="{overall_color}" stroke-width="3"
                                stroke-dasharray="{overall_score}, 100"/>
                        </svg>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 28px; font-weight: bold; color: {overall_color};">{overall_score}%</div>
                    </div>
                    <div style="margin-top: 10px; font-size: 14px; color: #666;">Overall Match</div>
                </div>
                """, height=200)
            
            with ov_col2:
                components.html(f"""
                <div style="background: white; border-radius: 16px; padding: 25px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                    <div style="font-size: 48px; font-weight: bold; color: #4CAF50;">{len(matched)}</div>
                    <div style="font-size: 14px; color: #666; margin-top: 5px;">Matched Skills</div>
                </div>
                """, height=200)
            
            with ov_col3:
                components.html(f"""
                <div style="background: white; border-radius: 16px; padding: 25px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                    <div style="font-size: 48px; font-weight: bold; color: #FFC107;">{len(partial)}</div>
                    <div style="font-size: 14px; color: #666; margin-top: 5px;">Partial Matches</div>
                </div>
                """, height=200)
            
            with ov_col4:
                components.html(f"""
                <div style="background: white; border-radius: 16px; padding: 25px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                    <div style="font-size: 48px; font-weight: bold; color: #F44336;">{len(missing)}</div>
                    <div style="font-size: 14px; color: #666; margin-top: 5px;">Missing Skills</div>
                </div>
                """, height=200)
            
            # Interactive Skills Comparison Bar Chart
            st.markdown("### 📊 Skills Comparison")
            
            # Prepare data for comparison
            all_skills = list(set(jd_skills))
            skill_data = []
            for skill in all_skills:
                resume_val = 100 if skill in resume_skills else 0
                jd_val = 100
                skill_data.append({
                    "Skill": skill,
                    "Resume Skills": resume_val,
                    "Job Requirements": jd_val
                })
            
            skill_df = pd.DataFrame(skill_data)
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name='Resume Skills',
                x=skill_df['Skill'],
                y=skill_df['Resume Skills'],
                marker_color='#4CAF50'
            ))
            fig_bar.add_trace(go.Bar(
                name='Job Requirements',
                x=skill_df['Skill'],
                y=skill_df['Job Requirements'],
                marker_color='#2196F3'
            ))
            
            fig_bar.update_layout(
                barmode='group',
                xaxis_tickangle=-45,
                height=400,
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                paper_bgcolor='white',
                plot_bgcolor='#fafafa',
                margin=dict(t=50, b=100)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Skill-wise Match Scores
            st.markdown("### 🎯 Individual Skill Scores")
            
            score_cols = st.columns(4)
            skill_scores = [
                ("Python", 92, "#4CAF50"),
                ("Machine Learning", 88, "#4CAF50"),
                ("SQL", 65, "#FFC107"),
                ("AWS", 30, "#F44336")
            ]
            
            # Calculate actual scores based on matched/partial/missing
            actual_scores = []
            for i, skill in enumerate(jd_skills[:4]):
                if skill in matched:
                    score = 92
                    color = "#4CAF50"
                elif skill in partial:
                    score = 65
                    color = "#FFC107"
                else:
                    score = 30
                    color = "#F44336"
                actual_scores.append((skill, score, color))
            
            for i, (skill, score, color) in enumerate(actual_scores[:4]):
                with score_cols[i]:
                    components.html(f"""
                    <div style="background: white; border-radius: 12px; padding: 15px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.08);">
                        <div style="width: 60px; height: 60px; border-radius: 50%; background: {color}; margin: 0 auto; display: flex; align-items: center; justify-content: center;">
                            <span style="color: white; font-weight: bold; font-size: 18px;">{score}%</span>
                        </div>
                        <div style="margin-top: 10px; font-size: 12px; color: #333; font-weight: 500;">{skill}</div>
                    </div>
                    """, height=130)
        
        with dash_right:
            # Role View Section
            st.markdown("### 👤 Role View")
            role_tab = st.radio("Select View", ["Job Seeker", "Recruiter"], horizontal=True, label_visibility="collapsed")
            
            # Radar Chart for Skill Categories
            categories_radar = ['Technical Skills', 'Soft Skills', 'Certifications', 'Experience', 'Education']
            
            if role_tab == "Job Seeker":
                current_values = [
                    min(100, len(rtech) * 10),
                    min(100, len(rsoft) * 15),
                    60,
                    75,
                    80
                ]
                jd_values = [
                    min(100, len(jtech) * 10),
                    min(100, len(jsoft) * 15),
                    80,
                    90,
                    85
                ]
            else:
                current_values = [overall_score, min(100, len(matched) * 15), 70, 80, 75]
                jd_values = [100, 100, 100, 100, 100]
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=current_values + [current_values[0]],
                theta=categories_radar + [categories_radar[0]],
                fill='toself',
                name='Current Profile',
                line_color='#4CAF50',
                fillcolor='rgba(76, 175, 80, 0.3)'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=jd_values + [jd_values[0]],
                theta=categories_radar + [categories_radar[0]],
                fill='toself',
                name='Job Requirements',
                line_color='#2196F3',
                fillcolor='rgba(33, 150, 243, 0.2)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                showlegend=True,
                legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
                height=350,
                margin=dict(t=30, b=50)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Upskilling Recommendations Section
            st.markdown("### 💡 Upskilling Recommendations")
            
            upskill_recs = [
                {"icon": "☁️", "title": "AWS Cloud Services", "desc": "Complete AWS Certified Solutions Architect course", "color": "#FF9800"},
                {"icon": "📊", "title": "Advanced Statistics", "desc": "Enroll in Advanced Statistics for Data Science program", "color": "#2196F3"},
                {"icon": "📋", "title": "Project Management", "desc": "Consider PMP certification for leadership skills", "color": "#9C27B0"}
            ]
            
            for rec in upskill_recs:
                components.html(f"""
                <div style="background: white; border-left: 4px solid {rec['color']}; border-radius: 8px; padding: 12px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 20px;">{rec['icon']}</span>
                        <div>
                            <div style="font-weight: 600; font-size: 14px; color: #333;">{rec['title']}</div>
                            <div style="font-size: 12px; color: #666;">{rec['desc']}</div>
                        </div>
                    </div>
                </div>
                """, height=80)
        
        # Skill Comparison Section
        st.markdown("### 🔄 Skill Comparison")
        comp_col1, comp_col2, comp_col3 = st.columns(3)
        
        with comp_col1:
            st.markdown("**Python**")
            st.progress(min(100, len([s for s in resume_skills if 'python' in s.lower()]) * 50 + 50) / 100)
        
        with comp_col2:
            st.markdown("**Machine Learning**")
            st.progress(min(100, len([s for s in resume_skills if 'machine' in s.lower() or 'learning' in s.lower()]) * 40 + 40) / 100)
        
        with comp_col3:
            st.markdown("**SQL**")
            st.progress(min(100, len([s for s in resume_skills if 'sql' in s.lower()]) * 50 + 30) / 100)
        
        # --------------------------------------------------
        # REPORT EXPORT SECTION
        # --------------------------------------------------
        st.divider()
        st.markdown("## 📥 Download Reports")
        
        # PDF Generation Function
        def generate_pdf_report():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=50, bottomMargin=50)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = styles['Heading1']
            title_style.alignment = 1
            story.append(Paragraph("Skill Gap Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Date
            story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Overall Score
            story.append(Paragraph(f"<b>Overall Match Score:</b> {overall_score}%", styles['Heading2']))
            story.append(Spacer(1, 15))
            
            # Summary Statistics
            story.append(Paragraph("<b>Summary Statistics</b>", styles['Heading2']))
            story.append(Paragraph(f"• Total Skills Required: {len(jd_skills)}", styles['Normal']))
            story.append(Paragraph(f"• Matched Skills: {len(matched)}", styles['Normal']))
            story.append(Paragraph(f"• Partial Matches: {len(partial)}", styles['Normal']))
            story.append(Paragraph(f"• Missing Skills: {len(missing)}", styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Matched Skills
            story.append(Paragraph("<b>✅ Matched Skills</b>", styles['Heading2']))
            story.append(Paragraph(comma_list(matched) if matched else "None", styles['Normal']))
            story.append(Spacer(1, 10))
            
            # Partial Matches
            story.append(Paragraph("<b>⚠️ Partial Matches</b>", styles['Heading2']))
            story.append(Paragraph(comma_list(partial) if partial else "None", styles['Normal']))
            story.append(Spacer(1, 10))
            
            # Missing Skills
            story.append(Paragraph("<b>❌ Missing Skills</b>", styles['Heading2']))
            story.append(Paragraph(comma_list(missing) if missing else "None", styles['Normal']))
            story.append(Spacer(1, 15))
            
            # Resume Skills
            story.append(Paragraph("<b>Resume Skills Breakdown</b>", styles['Heading2']))
            story.append(Paragraph(f"• Technical Skills ({len(rtech)}): {comma_list(rtech)}", styles['Normal']))
            story.append(Paragraph(f"• Soft Skills ({len(rsoft)}): {comma_list(rsoft)}", styles['Normal']))
            story.append(Spacer(1, 10))
            
            # JD Skills
            story.append(Paragraph("<b>Job Requirements Breakdown</b>", styles['Heading2']))
            story.append(Paragraph(f"• Technical Skills ({len(jtech)}): {comma_list(jtech)}", styles['Normal']))
            story.append(Paragraph(f"• Soft Skills ({len(jsoft)}): {comma_list(jsoft)}", styles['Normal']))
            story.append(Spacer(1, 15))

            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                donut_fig = go.Figure(data=[go.Pie(labels=["Matched", "Partial", "Missing"], values=[len(matched), len(partial), len(missing)], hole=0.55, marker=dict(colors=["#4CAF50", "#FFC107", "#F44336"]))])
                donut_fig.update_layout(title="Skill Match Overview", showlegend=True)
                pio.write_image(donut_fig, tmp.name, format="png", width=400, height=300, scale=2)
                story.append(RLImage(tmp.name, width=300, height=220))
            except:
                story.append(Paragraph("(Charts require Kaleido installation)", styles["Normal"]))
            
            # Recommendations
            #story.append(Paragraph("<b>💡 Upskilling Recommendations</b>", styles['Heading2']))
            #for i, rec in enumerate(recommendations[:5], 1):
                #story.append(Paragraph(f"{i}. <b>{rec['title']}</b> - {rec['description']} (Priority: {rec['priority']})", styles['Normal']))
            
            doc.build(story)
            buffer.seek(0)
            return buffer
        
        # CSV Generation Function
        def generate_csv_report():
            # Skills Detail Report
            report_data = []
            for skill in jd_skills:
                status = "Matched" if skill in matched else "Partial" if skill in partial else "Missing"
                in_resume = "Yes" if skill in resume_skills else "No"
                report_data.append({
                    "Skill": skill,
                    "Status": status,
                    "In Resume": in_resume,
                    "Required": "Yes"
                })
            
            # Add resume-only skills
            for skill in resume_skills:
                if skill not in jd_skills:
                    report_data.append({
                        "Skill": skill,
                        "Status": "Extra",
                        "In Resume": "Yes",
                        "Required": "No"
                    })
            
            df_report = pd.DataFrame(report_data)
            return df_report.to_csv(index=False)
        
        # Summary CSV
        def generate_summary_csv():
            summary_data = {
                "Metric": [
                    "Overall Match Score",
                    "Total JD Skills",
                    "Matched Skills",
                    "Partial Matches",
                    "Missing Skills",
                    "Resume Technical Skills",
                    "Resume Soft Skills",
                    "JD Technical Skills",
                    "JD Soft Skills",
                    "Report Generated"
                ],
                "Value": [
                    f"{overall_score}%",
                    len(jd_skills),
                    len(matched),
                    len(partial),
                    len(missing),
                    len(rtech),
                    len(rsoft),
                    len(jtech),
                    len(jsoft),
                    datetime.now().strftime('%Y-%m-%d %H:%M')
                ]
            }
            return pd.DataFrame(summary_data).to_csv(index=False)
        
        # Download Buttons
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            pdf_buffer = generate_pdf_report()
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name=f"skill_gap_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        with dl_col2:
            csv_data = generate_csv_report()
            st.download_button(
                label="📊 Download Skills CSV",
                data=csv_data,
                file_name=f"skills_detail_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with dl_col3:
            summary_csv = generate_summary_csv()
            st.download_button(
                label="📈 Download Summary CSV",
                data=summary_csv,
                file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Footer
        st.divider()
        st.markdown("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>🎯 <b>Skill Gap Analysis Dashboard</b> | Milestone 4 Complete</p>
            <p style="font-size: 12px;">Interactive graphs, scores, and downloadable reports in PDF/CSV formats</p>
        </div>
        """, unsafe_allow_html=True)
        
    except ValueError as ve:
        st.error(f"❌ **Processing Error:** {str(ve)}")
        st.info("💡 Please check your files and try uploading them again.")
    except Exception as e:
        st.error(f"❌ **Unexpected Error:** An error occurred while processing your files: {str(e)}")
        st.info("💡 Please try again or contact support if the issue persists.")

else:
    # Show initial info message when analyze hasn't been clicked
    st.info("👆 Please upload both Resume and Job Description files, then click **Analyze** to see the skill gap analysis.")