import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR

# Page config
st.set_page_config(
    page_title="ระบบทำนายประสิทธิภาพการนอนหลับ",
    page_icon="",
    layout="wide"
)

# ====== TRAIN MODEL FUNCTION ======
@st.cache_resource
def train_and_load_model():
    """Train โมเดลและ return model + preprocessor"""
    
    # ตรวจสอบว่ามีไฟล์ CSV ไหม
    csv_path = Path('data/Sleep_Efficiency.csv')
    if not csv_path.exists():
        # ลอง path อื่นๆ
        possible_paths = [
            'Sleep_Efficiency.csv',
            './Sleep_Efficiency.csv',
            '../data/Sleep_Efficiency.csv'
        ]
        for p in possible_paths:
            if Path(p).exists():
                csv_path = Path(p)
                break
        else:
            st.error("❌ ไม่พบไฟล์ข้อมูล Sleep_Efficiency.csv")
            st.stop()
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Feature Engineering
    df['Bedtime'] = pd.to_datetime(df['Bedtime'])
    df['Wakeup time'] = pd.to_datetime(df['Wakeup time'])
    
    df['Bedtime_hour'] = df['Bedtime'].dt.hour + df['Bedtime'].dt.minute / 60
    df['Wakeup_hour'] = df['Wakeup time'].dt.hour + df['Wakeup time'].dt.minute / 60
    df['Sleep_duration_calc'] = (df['Wakeup time'] - df['Bedtime']).dt.total_seconds() / 3600
    df.loc[df['Sleep_duration_calc'] < 0, 'Sleep_duration_calc'] += 24
    
    df = df.drop(columns=['ID', 'Bedtime', 'Wakeup time', 'Sleep duration'])
    
    X = df.drop(columns=['Sleep efficiency'])
    y = df['Sleep efficiency']
    
    # Define features
    numeric_features = ['Age', 'REM sleep percentage', 'Deep sleep percentage',
                        'Light sleep percentage', 'Awakenings', 'Caffeine consumption',
                        'Alcohol consumption', 'Exercise frequency',
                        'Bedtime_hour', 'Wakeup_hour', 'Sleep_duration_calc']
    
    categorical_features = ['Gender', 'Smoking status']
    
    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model (ใช้พารามิเตอร์ที่ดีที่สุด)
    svm_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('svr', SVR(C=10, epsilon=0.01, kernel='rbf', gamma='scale'))
    ])
    
    svm_pipeline.fit(X_train, y_train)
    
    return svm_pipeline

# ====== LOAD MODEL ======
try:
    model = train_and_load_model()
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในการโหลด/ฝึกโมเดล: {str(e)}")
    st.stop()

# Header
st.title("😴 ระบบทำนายประสิทธิภาพการนอนหลับ")
st.markdown("### 🎯 ทำนายคุณภาพการนอนของคุณด้วยเทคโนโลยี Machine Learning")
st.markdown("---")

# Sidebar - Input Form
st.sidebar.header("📝 กรอกข้อมูลการนอนหลับ")

with st.sidebar.form("input_form"):
    st.subheader("ข้อมูลส่วนบุคคล")
    age = st.number_input("อายุ (ปี)", min_value=1, max_value=120, value=35, help="ระบุอายุของคุณ")
    gender = st.selectbox("เพศ", ["ชาย", "หญิง"], help="เลือกเพศของคุณ")
    
    st.subheader("เวลาการนอน")
    col1, col2 = st.columns(2)
    with col1:
        bedtime_hour = st.slider("ชั่วโมงเข้านอน", 0, 23, 23)
        bedtime_min = st.slider("นาทีเข้านอน", 0, 59, 0)
    with col2:
        wakeup_hour = st.slider("ชั่วโมงตื่นนอน", 0, 23, 7)
        wakeup_min = st.slider("นาทีตื่นนอน", 0, 59, 0)
    
    st.subheader("ระยะการนอน")
    rem = st.slider("ระยะ REM (%)", 0, 100, 20, help="เปอร์เซ็นต์การนอนระยะ REM")
    deep = st.slider("ระยะหลับลึก (%)", 0, 100, 50, help="เปอร์เซ็นต์การนอนระยะหลับลึก")
    light = st.slider("ระยะหลับตื้น (%)", 0, 100, 30, help="เปอร์เซ็นต์การนอนระยะหลับตื้น")
    
    st.subheader("พฤติกรรมและไลฟ์สไตล์")
    awakenings = st.number_input("จำนวนครั้งที่ตื่นกลางคืน", 0, 20, 1, help="จำนวนครั้งที่ตื่นขึ้นกลางคืน")
    caffeine = st.number_input("คาเฟอีน (มก./วัน)", 0, 500, 50, help="ปริมาณคาเฟอีนที่บริโภคต่อวัน")
    alcohol = st.number_input("แอลกอฮอล์ (แก้ว/วัน)", 0, 20, 0, help="จำนวนแก้วแอลกอฮอล์ต่อวัน")
    smoking = st.selectbox("สูบบุหรี่", ["ใช่", "ไม่"], help="สถานะการสูบบุหรี่")
    exercise = st.slider("ความถี่ในการออกกำลังกาย (วัน/สัปดาห์)", 0, 7, 3, help="จำนวนวันที่ออกกำลังกายต่อสัปดาห์")
    
    submitted = st.form_submit_button("🔮 เริ่มทำนาย", use_container_width=True)

# Calculate features
def calculate_features():
    bedtime_total = bedtime_hour + bedtime_min / 60
    wakeup_total = wakeup_hour + wakeup_min / 60
    
    duration = wakeup_total - bedtime_total
    if duration < 0:
        duration += 24
    
    # แปลงเพศและสูบบุหรี่เป็นภาษาอังกฤษ (ตามข้อมูลต้นฉบับ)
    gender_eng = "Male" if gender == "ชาย" else "Female"
    smoking_eng = "Yes" if smoking == "ใช่" else "No"
    
    data = {
        'Age': [age],
        'REM sleep percentage': [rem],
        'Deep sleep percentage': [deep],
        'Light sleep percentage': [light],
        'Awakenings': [awakenings],
        'Caffeine consumption': [caffeine],
        'Alcohol consumption': [alcohol],
        'Exercise frequency': [exercise],
        'Bedtime_hour': [bedtime_total],
        'Wakeup_hour': [wakeup_total],
        'Sleep_duration_calc': [duration],
        'Gender': [gender_eng],
        'Smoking status': [smoking_eng]
    }
    return pd.DataFrame(data)

# Predict
if submitted:
    input_df = calculate_features()
    prediction = model.predict(input_df)[0]
    
    st.markdown("---")
    st.subheader(" ผลการทำนาย")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 ประสิทธิภาพการนอนหลับ",
            value=f"{prediction:.2%}",
            delta=None
        )
    
    with col2:
        if prediction >= 0.85:
            level = " ดีมาก"
            level_eng = "Excellent"
        elif prediction >= 0.70:
            level = "🟡 ปานกลาง"
            level_eng = "Moderate"
        else:
            level = "🔴 ควรปรับปรุง"
            level_eng = "Needs Improvement"
        st.metric(label="📊 ระดับคุณภาพ", value=level)
    
    with col3:
        duration_hours = input_df['Sleep_duration_calc'][0]
        st.metric(label="⏱️ ระยะเวลาการนอน", value=f"{duration_hours:.1f} ชม.")
    
    # Progress bar
    st.markdown("### 📈 ระดับคะแนน")
    st.progress(min(max(prediction, 0), 1))
    
    # คำแนะนำ
    st.markdown("---")
    st.subheader(" คำแนะนำเพื่อปรับปรุงการนอนหลับ")
    
    if prediction < 0.70:
        st.error("### ⚠️ การนอนหลับของคุณต้องการการปรับปรุง")
        st.markdown("""
        **คำแนะนำ:**
        - 🚫 **ลดคาเฟอีนและแอลกอฮอล์** โดยเฉพาะก่อนนอน 4-6 ชั่วโมง
        - ⏰ **เข้านอนและตื่นนอนเวลาเดิม** ทุกวัน แม้แต่วันหยุด
        -  **ออกกำลังกายสม่ำเสมอ** แต่หลีกเลี่ยงก่อนนอน 2-3 ชั่วโมง
        - 📱 **หลีกเลี่ยงหน้าจอ** (มือถือ, คอมพิวเตอร์) 1 ชั่วโมงก่อนนอน
        - 🛏️ **สร้างสภาพแวดล้อมที่ดี** ห้องนอนควรมืด เงียบ และเย็นสบาย
        -  **ฝึกการผ่อนคลาย** เช่น การหายใจลึกๆ หรือการทำสมาธิก่อนนอน
        """)
    elif prediction < 0.85:
        st.warning("### ⚡ การนอนหลับของคุณอยู่ในระดับปานกลาง")
        st.markdown("""
        **คำแนะนำ:**
        - พยายามรักษาเวลาเข้านอนให้สม่ำเสมอ
        - เพิ่มระยะเวลาการนอนหลับลึก (Deep Sleep)
        - ลดจำนวนครั้งการตื่นกลางคืน
        - ตรวจสอบปริมาณคาเฟอีนและแอลกอฮอล์
        """)
    else:
        st.success("### 🎉 ยินดีด้วย! การนอนหลับของคุณมีคุณภาพดีมาก")
        st.markdown("""
        **รักษามาตรฐานนี้ไว้:**
        - ✅ คุณมีพฤติกรรมการนอนที่ดี
        - ✅ รักษาเวลาการนอนที่สม่ำเสมอ
        - ✅ การออกกำลังกายเป็นประจำช่วยได้มาก
        - ✅ ควร继续保持这样的好习惯
        """)
    
    # แสดงรายละเอียดเพิ่มเติม
    st.markdown("---")
    st.subheader("📋 ข้อมูลที่คุณกรอก")
    st.json({
        "อายุ": f"{age} ปี",
        "เพศ": gender,
        "เวลาเข้านอน": f"{bedtime_hour:02d}:{bedtime_min:02d}",
        "เวลาตื่นนอน": f"{wakeup_hour:02d}:{wakeup_min:02d}",
        "ระยะ REM": f"{rem}%",
        "ระยะหลับลึก": f"{deep}%",
        "ระยะหลับตื้น": f"{light}%",
        "ตื่นกลางคืน": f"{awakenings} ครั้ง",
        "คาเฟอีน": f"{caffeine} มก.",
        "แอลกอฮอล์": f"{alcohol} แก้ว",
        "สูบบุหรี่": smoking,
        "ออกกำลังกาย": f"{exercise} วัน/สัปดาห์"
    })

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>พัฒนาด้วย ❤️ โดยใช้ Streamlit และ Machine Learning</p>
    <p>© 2024 Sleep Efficiency Predictor</p>
</div>
""", unsafe_allow_html=True)