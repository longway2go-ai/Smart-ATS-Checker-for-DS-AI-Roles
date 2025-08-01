import streamlit as st
import docx2txt
import json
import re
from pdfminer.high_level import extract_text as extract_pdf
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

# ---------- Skills Dictionary ----------
SKILLS_DATA = {
    "categories": {
        "languages": ["python", "r", "java", "c++", "sql", "html", "css", "javascript", "dax"],
        "libraries": [
            "pandas", "numpy", "scikit-learn", "tensorflow", "keras", "xgboost", "shap", "lime", "scipy",
            "lightgbm", "catboost", "nltk", "spacy", "plotly", "matplotlib", "seaborn", "ann", "cnn", 
            "rnn", "lstm", "mlp", "neural networks"
        ],
        "frameworks": [
            "flask", "fastapi", "streamlit", "gradio", "huggingface", "replit"
        ],
        "llm_frameworks": [
            "langchain", "transformers", "llamaindex", "openai", "langgraph",
            "langsmith", "mcp", "smolagents", "n8n", "sentence-transformers"
        ],
        "tools": [
            "jupyter", "kaggle", "vs code", "excel", "git", "github", "colab",
            "powerbi", "label studio", "google colab", "tableau"
        ],
        "mlops_llmops": [
            "mlflow", "dvc", "ray", "bentoml", "kubeflow", "weights & biases",
            "docker", "kubernetes", "vectordb", "pinecone", "faiss", "chromadb", "qdrantdb"
        ],
        "cloud": [
            "aws", "gcp", "azure", "sagemaker", "vertex ai", "azure ml", "google colab"
        ],
        "databases": ["mysql", "postgresql", "sqlite", "mongodb", "firebase"],
        "certifications": [
            "coursera", "udemy", "deeplearning.ai", "fast.ai", "aws certified", "google cloud certified"
        ]
    }
}

# ---------- Enhanced Role Definitions ----------
ROLE_REQUIREMENTS = {
    "Data Scientist": {
        "must_have": ["python", "pandas", "numpy", "scikit-learn", "sql"],
        "preferred": ["matplotlib", "seaborn", "jupyter", "statistics", "plotly"],
        "nice_to_have": ["r", "tensorflow", "keras", "aws", "git"],
        "weight_multiplier": {"libraries": 2.0, "languages": 1.5, "tools": 1.2},
        "description": "Focuses on statistical analysis, data exploration, and building predictive models"
    },
    "ML Engineer": {
        "must_have": ["python", "scikit-learn", "git", "docker"],
        "preferred": ["tensorflow", "pytorch", "mlflow", "kubernetes", "aws"],
        "nice_to_have": ["xgboost", "lightgbm", "airflow", "bentoml"],
        "weight_multiplier": {"mlops_llmops": 2.5, "cloud": 2.0, "libraries": 1.8},
        "description": "Deploys and maintains ML models in production environments"
    },
    "LLM Engineer": {
        "must_have": ["python", "transformers", "langchain"],
        "preferred": ["huggingface", "openai", "llamaindex", "vectordb"],
        "nice_to_have": ["pinecone", "chromadb", "docker", "aws"],
        "weight_multiplier": {"llm_frameworks": 3.0, "frameworks": 2.0, "cloud": 1.5},
        "description": "Specializes in large language models and generative AI applications"
    },
    "AI Engineer": {
        "must_have": ["python", "tensorflow", "neural networks"],
        "preferred": ["pytorch", "deep learning", "computer vision", "docker"],
        "nice_to_have": ["reinforcement learning", "kubernetes", "aws", "mlflow"],
        "weight_multiplier": {"libraries": 2.5, "mlops_llmops": 2.0, "cloud": 1.8},
        "description": "Develops AI systems and neural network architectures"
    },
    "GenAI Engineer": {
        "must_have": ["python", "transformers", "langchain"],
        "preferred": ["llamaindex", "vectordb", "huggingface", "gradio"],
        "nice_to_have": ["pinecone", "chromadb", "streamlit", "fastapi"],
        "weight_multiplier": {"llm_frameworks": 3.0, "frameworks": 2.2, "libraries": 1.8},
        "description": "Creates generative AI applications and conversational systems"
    },
    "AI/ML Developer": {
        "must_have": ["python", "flask", "git"],
        "preferred": ["fastapi", "streamlit", "docker", "scikit-learn"],
        "nice_to_have": ["bentoml", "aws", "kubernetes", "gradio"],
        "weight_multiplier": {"frameworks": 2.5, "tools": 2.0, "cloud": 1.8},
        "description": "Builds applications and APIs for AI/ML solutions"
    }
}

# ---------- Text Extraction ----------
def extract_text(file):
    try:
        if file.name.endswith(".pdf"):
            return extract_pdf(file)
        elif file.name.endswith(".docx"):
            return docx2txt.process(file)
        else:
            return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return ""

# ---------- Keyword Matching ----------
def find_skills_in_text(text, skills_dict):
    matches = {category: [] for category in skills_dict}
    text_lower = text.lower()
    
    for category, keywords in skills_dict.items():
        for keyword in keywords:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text_lower):
                matches[category].append(keyword)
    
    return matches

# ---------- Role Scoring ----------
def calculate_role_fit(resume_skills, role_name, role_req):
    score_breakdown = {
        "must_have_score": 0,
        "preferred_score": 0,
        "nice_to_have_score": 0,
        "total_skills": 0,
        "missing_critical": []
    }
    
    # Flatten all resume skills
    all_resume_skills = []
    for category, skills in resume_skills.items():
        all_resume_skills.extend([skill.lower() for skill in skills])
    
    # Check must-have skills
    must_have_found = 0
    for skill in role_req["must_have"]:
        if skill.lower() in all_resume_skills:
            must_have_found += 1
        else:
            score_breakdown["missing_critical"].append(skill)
    
    score_breakdown["must_have_score"] = (must_have_found / len(role_req["must_have"])) * 100
    
    # Check preferred skills
    preferred_found = sum(1 for skill in role_req["preferred"] if skill.lower() in all_resume_skills)
    score_breakdown["preferred_score"] = (preferred_found / len(role_req["preferred"])) * 100
    
    # Check nice-to-have skills
    nice_found = sum(1 for skill in role_req["nice_to_have"] if skill.lower() in all_resume_skills)
    score_breakdown["nice_to_have_score"] = (nice_found / len(role_req["nice_to_have"])) * 100
    
    # Calculate weighted total score
    weights = role_req.get("weight_multiplier", {})
    weighted_score = 0
    total_possible = 0
    
    for category, skills in resume_skills.items():
        if skills:  # Only count categories where we found skills
            category_weight = weights.get(category, 1.0)
            weighted_score += len(set(skills)) * category_weight
            total_possible += 10 * category_weight  # Assume max 10 skills per category
    
    # Combine scores (weighted toward must-have skills)
    final_score = (
        score_breakdown["must_have_score"] * 0.5 +
        score_breakdown["preferred_score"] * 0.3 +
        score_breakdown["nice_to_have_score"] * 0.2
    )
    
    score_breakdown["final_score"] = min(final_score, 100)  # Cap at 100
    score_breakdown["total_skills"] = len(all_resume_skills)
    
    return score_breakdown

# ---------- Streamlit App ----------
st.set_page_config(page_title="Smart ATS Checker", layout="wide", page_icon="🎯")

st.title("🎯 Smart ATS Checker for DS/AI Roles")
st.markdown("**Upload your resume and instantly discover which AI/ML roles match your skillset best!**")

# Sidebar with role information
with st.sidebar:
    st.header("📋 Available Roles")
    for role, req in ROLE_REQUIREMENTS.items():
        with st.expander(f"🔹 {role}"):
            st.write(req["description"])
            st.write("**Must have:**", ", ".join(req["must_have"][:3]) + "...")

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📄 Upload Resume")
    resume_file = st.file_uploader(
        "Choose your resume file", 
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT"
    )
    
    if resume_file:
        st.success(f"✅ {resume_file.name} uploaded successfully!")

with col2:
    st.subheader("🎯 Analysis Options")
    
    analysis_type = st.radio(
        "Choose analysis type:",
        ["🚀 Quick Role Match", "📊 Detailed Breakdown", "🔍 Custom Job Description"],
        help="Quick match shows top roles, detailed shows all metrics"
    )
    
    custom_job_desc = ""
    if analysis_type == "🔍 Custom Job Description":
        custom_job_desc = st.text_area(
            "Paste job description here:",
            height=150,
            placeholder="Paste the job description to get a custom match score..."
        )

if st.button("🔍 Analyze My Resume", type="primary", use_container_width=True):
    if not resume_file:
        st.error("Please upload your resume first!")
    else:
        with st.spinner("🔄 Analyzing your resume..."):
            # Extract text from resume
            resume_text = extract_text(resume_file)
            
            if not resume_text.strip():
                st.error("Could not extract text from the resume. Please check the file format.")
            else:
                # Find skills in resume
                skills_dict = SKILLS_DATA["categories"]
                resume_skills = find_skills_in_text(resume_text, skills_dict)
                
                # Calculate scores for all roles
                role_scores = {}
                for role_name, role_req in ROLE_REQUIREMENTS.items():
                    role_scores[role_name] = calculate_role_fit(resume_skills, role_name, role_req)
                
                # Sort roles by score
                sorted_roles = sorted(role_scores.items(), key=lambda x: x[1]["final_score"], reverse=True)
                
                if analysis_type == "🚀 Quick Role Match":
                    # Quick overview
                    st.success("✅ Analysis Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        best_role = sorted_roles[0]
                        st.metric(
                            "🏆 Best Match", 
                            best_role[0], 
                            f"{best_role[1]['final_score']:.1f}%"
                        )
                    
                    with col2:
                        total_skills = sum(len(skills) for skills in resume_skills.values() if skills)
                        st.metric("🔧 Total Skills Found", total_skills)
                    
                    with col3:
                        avg_score = sum(score["final_score"] for _, score in sorted_roles) / len(sorted_roles)
                        st.metric("📊 Average Match", f"{avg_score:.1f}%")
                    
                    # Top 3 roles
                    st.subheader("🎯 Top Role Matches")
                    for i, (role, score_data) in enumerate(sorted_roles[:3]):
                        with st.expander(f"#{i+1} {role} - {score_data['final_score']:.1f}% match", expanded=(i==0)):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Description:** {ROLE_REQUIREMENTS[role]['description']}")
                                st.write(f"**Must-have skills match:** {score_data['must_have_score']:.1f}%")
                                st.write(f"**Preferred skills match:** {score_data['preferred_score']:.1f}%")
                            
                            with col2:
                                if score_data['missing_critical']:
                                    st.write("**⚠️ Missing critical skills:**")
                                    for skill in score_data['missing_critical'][:5]:
                                        st.write(f"• {skill}")
                                else:
                                    st.write("✅ **All critical skills found!**")
                
                elif analysis_type == "📊 Detailed Breakdown":
                    # Detailed analysis
                    st.success("✅ Detailed Analysis Complete!")
                    
                    # Role comparison chart
                    fig = go.Figure()
                    roles = [role for role, _ in sorted_roles]
                    scores = [score_data["final_score"] for _, score_data in sorted_roles]
                    
                    fig.add_trace(go.Bar(
                        x=roles,
                        y=scores,
                        marker_color=['#FF6B6B' if i == 0 else '#4ECDC4' for i in range(len(roles))],
                        text=[f"{score:.1f}%" for score in scores],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title="Role Match Scores Comparison",
                        xaxis_title="Roles",
                        yaxis_title="Match Score (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Skills breakdown
                    st.subheader("🔧 Skills Found in Your Resume")
                    skills_cols = st.columns(3)
                    
                    col_idx = 0
                    for category, skills in resume_skills.items():
                        if skills:
                            with skills_cols[col_idx % 3]:
                                st.write(f"**{category.replace('_', ' ').title()}** ({len(skills)})")
                                st.write(", ".join(sorted(set(skills))))
                                col_idx += 1
                    
                    # Detailed role analysis
                    st.subheader("📋 Detailed Role Analysis")
                    for role, score_data in sorted_roles:
                        with st.expander(f"{role} - {score_data['final_score']:.1f}% match"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Skill Category Breakdown:**")
                                st.write(f"• Must-have: {score_data['must_have_score']:.1f}%")
                                st.write(f"• Preferred: {score_data['preferred_score']:.1f}%")
                                st.write(f"• Nice-to-have: {score_data['nice_to_have_score']:.1f}%")
                            
                            with col2:
                                if score_data['missing_critical']:
                                    st.write("**Missing Critical Skills:**")
                                    for skill in score_data['missing_critical']:
                                        st.write(f"• {skill}")
                                else:
                                    st.success("All critical skills found!")
                
                else:  # Custom job description
                    if not custom_job_desc.strip():
                        st.warning("Please paste a job description for custom analysis.")
                    else:
                        # Analyze custom job description
                        job_skills = find_skills_in_text(custom_job_desc, skills_dict)
                        
                        # Calculate match
                        matched_skills = {}
                        total_job_skills = 0
                        total_matched = 0
                        
                        for category in skills_dict:
                            job_cat_skills = set(job_skills[category])
                            resume_cat_skills = set(resume_skills[category])
                            matched_skills[category] = list(job_cat_skills & resume_cat_skills)
                            
                            total_job_skills += len(job_cat_skills)
                            total_matched += len(matched_skills[category])
                        
                        match_percentage = (total_matched / total_job_skills * 100) if total_job_skills > 0 else 0
                        
                        st.success("✅ Custom Job Analysis Complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 Overall Match", f"{match_percentage:.1f}%")
                        with col2:
                            st.metric("✅ Skills Matched", total_matched)
                        with col3:
                            st.metric("📋 Job Requirements", total_job_skills)
                        
                        # Show matched skills
                        st.subheader("🎯 Matched Skills by Category")
                        for category, skills in matched_skills.items():
                            if skills:
                                st.write(f"**{category.replace('_', ' ').title()}:** {', '.join(skills)}")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Keep your resume updated with relevant keywords and consider adding skills from your target role's 'missing critical skills' list!")