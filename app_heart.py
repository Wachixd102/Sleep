import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ทำนายโรคหัวใจ",
    page_icon="❤️",
    layout="wide"
)

# โหลด CSS แบบเรียบหรู (ใช้ชุดเดียวกับงานก่อนหน้า)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');
* { font-family: 'Kanit', sans-serif; }
.stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%); background-attachment: fixed; }
.main-header { text-align: center; padding: 2rem 0; margin-bottom: 1.5rem; background: linear-gradient(120deg, #e74a3b 0%, #c83a30 100%); border-radius: 1rem; box-shadow: 0 4px 20px rgba(231, 74, 59, 0.3); color: white; }
.header-title { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }
.header-subtitle { font-size: 1.2rem; opacity: 0.9; }
.input-card { background: white; border-radius: 1rem; padding: 1.5rem; box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15); margin-bottom: 1rem; }
.section-title { font-size: 1.3rem; font-weight: 600; color: #e74a3b; margin-bottom: 1rem; display: flex; align-items: center; }
.section-title::before { content: ""; display: inline-block; width: 6px; height: 20px; background: #e74a3b; border-radius: 3px; margin-right: 10px; }
.result-card { background: white; border-radius: 1rem; padding: 2rem; box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15); text-align: center; }
.stButton > button { width: 100%; height: 50px; font-size: 1.1rem; font-weight: 600; border-radius: 1rem; background: linear-gradient(120deg, #e74a3b 0%, #c83a30 100%); color: white; border: none; transition: all 0.3s ease; box-shadow: 0 4px 10px rgba(231, 74, 59, 0.3); }
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(231, 74, 59, 0.4); }
</style>
""", unsafe_allow_html=True)

# ====== 1. โหลดและเตรียมข้อมูล ======
@st.cache_data
def load_data():
    df = pd.read_csv('Heart3.csv')
    return df

df = load_data()

# แยก Features (X) และ Target (y)
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# แบ่งข้อมูล Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== 2. สร้างและเทรนโมเดล ======
@st.cache_resource
def train_model():
    # Decision Tree Classifier (ตั้งค่า max_depth เพื่อป้องกัน Overfitting)
    model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# คำนวณความแม่นยำเพื่อแสดง
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ====== 3. ส่วนของ UI ======
st.markdown("""
<div class="main-header">
    <h1 class="header-title">❤️ ระบบทำนายความเสี่ยงโรคหัวใจ</h1>
    <p class="header-subtitle">วิเคราะห์ความเสี่ยงด้วยอัลกอริทึม Decision Tree</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">ข้อมูลสุขภาพพื้นฐาน</h3>', unsafe_allow_html=True)
    
    age = st.number_input("อายุ (Age)", min_value=20, max_value=100, value=50, step=1)
    sex = st.selectbox("เพศ (Sex)", options=[0, 1], format_func=lambda x: "หญิง (Female)" if x == 0 else "ชาย (Male)")
    cp = st.selectbox("ประเภทอาการเจ็บหน้าอก (ChestPainType)", options=[1, 2, 3, 4], format_func=lambda x: f"Type {x}")
    
    st.markdown('<h3 class="section-title" style="margin-top:1rem;">ผลการตรวจร่างกาย</h3>', unsafe_allow_html=True)
    
    trestbps = st.number_input("ความดันโลหิตขณะพัก (RestingBP)", min_value=80, max_value=200, value=130, step=1)
    chol = st.number_input("ระดับคอเลสเตอรอล (Cholesterol)", min_value=100, max_value=600, value=250, step=1)
    fbs = st.selectbox("น้ำตาลในเลือดขณะอดอาหาร > 120 (FastingBS)", options=[0, 1], format_func=lambda x: "ไม่ (No)" if x == 0 else "ใช่ (Yes)")
    restecg = st.selectbox("ผล ECG ขณะพัก (RestingECG)", options=[0, 1, 2], format_func=lambda x: f"Level {x}")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">ผลการทดสอบและพฤติกรรม</h3>', unsafe_allow_html=True)
    
    thalach = st.number_input("อัตราการเต้นของหัวใจสูงสุด (MaxHR)", min_value=60, max_value=220, value=150, step=1)
    exang = st.selectbox("มีอาการเจ็บหน้าอกเมื่อออกกำลังกาย (ExerciseAngina)", options=[0, 1], format_func=lambda x: "ไม่ (No)" if x == 0 else "ใช่ (Yes)")
    oldpeak = st.number_input("ค่า ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
    slope = st.selectbox("ความชันของ ST Segment (ST_Slope)", options=[1, 2, 3], format_func=lambda x: f"Level {x}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ปุ่มทำนาย
    st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)
    predict_btn = st.button("🔍 วิเคราะห์ความเสี่ยงโรคหัวใจ")
    st.markdown('</div>', unsafe_allow_html=True)

# ====== 4. แสดงผลการทำนาย ======
if predict_btn:
    # สร้าง DataFrame สำหรับ input
    input_data = pd.DataFrame({
        'Age': [age], 'Sex': [sex], 'ChestPainType': [cp],
        'RestingBP': [trestbps], 'Cholesterol': [chol], 'FastingBS': [fbs],
        'RestingECG': [restecg], 'MaxHR': [thalach], 'ExerciseAngina': [exang],
        'Oldpeak': [oldpeak], 'ST_Slope': [slope]
    })
    
    # ทำนายผล
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    prob_no = probabilities[0] * 100
    prob_yes = probabilities[1] * 100
    
    st.markdown("---")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if prediction == 1:
            st.markdown(f"""
            <h2 style="color: #e74a3b; font-size: 2.5rem;">⚠️ มีความเสี่ยง</h2>
            <p style="font-size: 1.2rem; color: #5a5c69;">มีโอกาสเป็นโรคหัวใจ <b>{prob_yes:.1f}%</b></p>
            <p style="color: #e74a3b;">ควรปรึกษาแพทย์เพื่อตรวจวินิจฉัยเพิ่มเติม</p>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <h2 style="color: #1cc88a; font-size: 2.5rem;">✅ ความเสี่ยงต่ำ</h2>
            <p style="font-size: 1.2rem; color: #5a5c69;">มีโอกาสเป็นโรคหัวใจเพียง <b>{prob_yes:.1f}%</b></p>
            <p style="color: #1cc88a;">สุขภาพหัวใจของคุณอยู่ในเกณฑ์ที่ดี</p>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title" style="justify-content:center;">📊 ปัจจัยที่มีผลต่อการทำนาย (Feature Importance)</h3>', unsafe_allow_html=True)
        
        # แสดงกราฟ Feature Importance
        importances = model.feature_importances_
        feature_names = X.columns
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=importances, y=feature_names, palette='viridis', ax=ax)
        ax.set_title('ความสำคัญของแต่ละปัจจัย (Decision Tree)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Features')
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# แสดงข้อมูล Model Performance
with st.expander("📈 ข้อมูลประสิทธิภาพของโมเดล (Model Performance)"):
    st.write(f"**ความแม่นยำ (Accuracy):** {accuracy * 100:.2f}%")
    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred))