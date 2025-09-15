# app/streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import json
import os
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from io import BytesIO
import base64

# ===== CONFIG =====
MODEL_PATH = "models/student_performance_pipeline.joblib"
META_PATH = "artifacts/metadata.json"
FI_PATH = "artifacts/feature_importances.csv"

# Configure Gemini API (set your key in Streamlit secrets or env var)
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 10px;
        margin-bottom: 30px;
    }
    .section-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 20px;
        margin-bottom: 15px;
        padding-left: 10px;
        border-left: 4px solid #1E88E5;
    }
    .info-text {
        background-color: #E3F2FD;
        padding: 12px;
        border-radius: 5px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 15px;
        font-size: 0.9rem;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 15px;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 20px;
    }
    .probability-table {
        width: 100%;
        text-align: center;
    }
    .probability-table thead th {
        background-color: #1E88E5;
        color: white;
        padding: 10px;
    }
    .feature-table {
        width: 100%;
    }
    .feature-table thead th {
        background-color: #5E35B1;
        color: white;
        padding: 10px;
    }
    .download-button {
        background-color: #1E88E5;
        color: white;
        padding: 12px 24px;
        border-radius: 5px;
        border: none;
        font-weight: bold;
        margin-top: 20px;
    }
    .ai-suggestions {
        background-color: #FFF8E1;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #FFC107;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_meta():
    with open(META_PATH, "r") as f:
        return json.load(f)

model = load_model()
meta = load_meta()

st.markdown('<h1 class="main-header">Student Performance Detection System</h1>', unsafe_allow_html=True)
st.markdown("Complete the student information form below to receive a performance prediction and personalized improvement recommendations.")

# Field descriptions dictionary
field_descriptions = {
    "age": "Student's current age in years (typically between 15-22 for secondary education).",
    "studytime": "Weekly study time: 1 (<2 hours), 2 (2-5 hours), 3 (5-10 hours), 4 (>10 hours).",
    "failures": "Number of past class failures (affects academic confidence and progression).",
    "absences": "Number of school absences (high absence rates correlate with lower performance).",
    "G1": "First period grade (scale 0-20, where 20 is the highest score).",
    "G2": "Second period grade (scale 0-20, where 20 is the highest score).",
    "school": "Student's school (different schools may have different resources and teaching methods).",
    "sex": "Student's gender (used for demographic analysis only).",
    "address": "Student's home address type (urban vs rural may affect access to resources).",
    "famsize": "Family size (may relate to available support and resources at home).",
    "Pstatus": "Parent's cohabitation status (can affect home stability and support).",
    "Medu": "Mother's education level (higher education often correlates with academic support).",
    "Fedu": "Father's education level (higher education often correlates with academic support).",
    "Mjob": "Mother's job (can affect available time and resources for academic support).",
    "Fjob": "Father's job (can affect available time and resources for academic support).",
    "reason": "Reason for choosing this school (proximity, reputation, course preference, etc.).",
    "guardian": "Student's primary guardian (may affect the type of support received).",
    "traveltime": "Home to school travel time: 1 (<15 min), 2 (15-30 min), 3 (30-60 min), 4 (>60 min).",
    "schoolsup": "Extra educational support from school (additional help can improve performance).",
    "famsup": "Family educational support (home support is crucial for academic success).",
    "paid": "Extra paid classes within the course subject (additional instruction can help).",
    "activities": "Extra-curricular activities (balance between academics and other activities).",
    "nursery": "Attended nursery school (early education can establish foundational skills).",
    "higher": "Wants to pursue higher education (aspiration can drive academic performance).",
    "internet": "Internet access at home (facilitates research and learning resources).",
    "romantic": "In a romantic relationship (can affect time allocation and emotional focus).",
    "famrel": "Quality of family relationships (scale 1-5, where 5 is excellent).",
    "freetime": "Free time after school (scale 1-5, where 5 is a lot of free time).",
    "goout": "Going out with friends (scale 1-5, where 5 is very frequently).",
    "Dalc": "Workday alcohol consumption (scale 1-5, where 5 is very high).",
    "Walc": "Weekend alcohol consumption (scale 1-5, where 5 is very high).",
    "health": "Current health status (scale 1-5, where 5 is very good)."
}

numeric_cols = meta["numeric"]
categorical_cols = [k for k in meta.keys() if k not in ("numeric", "target_classes")]

# === FORM WITH DESCRIPTIONS ===
with st.form("input_form"):
    st.markdown('<div class="section-header">Student Information</div>', unsafe_allow_html=True)
    
    input_data = {}
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Personal Details")
        input_data["age"] = st.number_input("Age", min_value=10, max_value=30, value=16,
                                            help=field_descriptions["age"])
        input_data["studytime"] = st.number_input("Study Time", min_value=1, max_value=4, value=2,
                                                  help=field_descriptions["studytime"])
        input_data["failures"] = st.number_input("Past Failures", min_value=0, max_value=5, value=0,
                                                 help=field_descriptions["failures"])
        input_data["absences"] = st.number_input("Absences", min_value=0, max_value=100, value=5,
                                                 help=field_descriptions["absences"])
        input_data["G1"] = st.number_input("First Period Grade (G1)", min_value=0, max_value=20, value=10,
                                           help=field_descriptions["G1"])
        input_data["G2"] = st.number_input("Second Period Grade (G2)", min_value=0, max_value=20, value=10,
                                           help=field_descriptions["G2"])
    
    with col2:
        st.markdown("#### Background Information")
        for col in categorical_cols:
            opts = meta[col]
            # Format the label with spaces between words
            label = ' '.join(word.capitalize() for word in col.split('_'))
            input_data[col] = st.selectbox(label, opts, index=0, 
                                          help=field_descriptions.get(col, f"Select {label}"))

    submitted = st.form_submit_button("Analyze Performance", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    classes = model.named_steps["clf"].classes_

    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
    st.markdown("### Performance Prediction")
    st.markdown(f"**Predicted Performance Level: {pred}**")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Prediction Details</div>', unsafe_allow_html=True)
    
    # Probabilities table with styling
    prob_df = pd.DataFrame({"Performance Level": classes, "Probability": proba})
    st.markdown("##### Prediction Confidence")
    st.table(prob_df.style.format({"Probability": "{:.2%}"})\
                .set_properties(**{'background-color': '#f0f2f6', 'color': 'black'})\
                .set_table_styles([{'selector': 'th', 'props': [('background-color', '#1E88E5'), 
                                                              ('color', 'white')]}]))
    
    # Feature importance
    st.markdown('<div class="section-header">Key Influencing Factors</div>', unsafe_allow_html=True)
    try:
        fi_df = pd.read_csv(FI_PATH)
        st.table(fi_df.style.format({"importance": "{:.4f}"})\
                    .set_properties(**{'background-color': '#f0f2f6', 'color': 'black'})\
                    .set_table_styles([{'selector': 'th', 'props': [('background-color', '#5E35B1'), 
                                                                  ('color', 'white')]}]))
    except Exception as e:
        st.error("Feature importance data is currently unavailable.")

    # === GEMINI AI TIPS ===
    st.markdown('<div class="section-header">Personalized Recommendations</div>', unsafe_allow_html=True)
    prompt = f"""
    The student's predicted performance is {pred}.
    Here are the student's characteristics: {input_data}.
    Provide specific, actionable recommendations to improve academic performance. 
    Format your response with clear headings and bullet points.
    Focus on practical strategies tailored to this student's situation.
    """
    try:
        model_ai = genai.GenerativeModel("gemini-1.5-flash")
        response = model_ai.generate_content(prompt)
        tips = response.text.strip()
        st.markdown('<div class="ai-suggestions">', unsafe_allow_html=True)
        st.markdown(tips)
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error("AI recommendations are temporarily unavailable. Please try again later.")
        tips = "AI suggestions could not be generated at this time."

    # === PDF REPORT GENERATION ===
    def create_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles for colored text
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            textColor=colors.HexColor('#1E88E5'),
            spaceAfter=30,
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            textColor=colors.HexColor('#0D47A1'),
            spaceAfter=12,
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading3'],
            textColor=colors.HexColor('#5E35B1'),
            spaceAfter=6,
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            textColor=colors.black,
            spaceAfter=6,
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Student Performance Analysis Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Prediction
        story.append(Paragraph("Performance Prediction", heading_style))
        story.append(Paragraph(f"Predicted Performance Level: <b>{pred}</b>", normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Probabilities
        story.append(Paragraph("Prediction Confidence", heading_style))
        prob_data = [["Performance Level", "Probability"]]
        for c, p in zip(classes, proba):
            prob_data.append([c, f"{p:.2%}"])
        
        prob_table = Table(prob_data)
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E88E5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f2f6')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Feature Importance
        story.append(Paragraph("Key Influencing Factors", heading_style))
        try:
            fi_df = pd.read_csv(FI_PATH)
            fi_data = [["Feature", "Importance"]]
            for _, row in fi_df.iterrows():
                fi_data.append([row['feature'], f"{row['importance']:.4f}"])
            
            fi_table = Table(fi_data)
            fi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#5E35B1')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f2f6')),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(fi_table)
        except:
            story.append(Paragraph("Feature importance data not available.", normal_style))
        story.append(Spacer(1, 0.2*inch))
        
        # AI Suggestions
        story.append(Paragraph("Personalized Recommendations", heading_style))
        # Format the AI response for better PDF presentation
        formatted_tips = tips.replace('**', '').replace('*', '• ')
        story.append(Paragraph(formatted_tips, normal_style))
        
        doc.build(story)
        pdf = buffer.getvalue()
        buffer.close()
        return pdf

    pdf_data = create_pdf()
    
    # Download button with icon
    st.download_button(
        label="Download Full Report (PDF)",
        data=pdf_data,
        file_name="student_performance_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )