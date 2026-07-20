import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR

# Page config
st.set_page_config(page_title="Sleep Efficiency Predictor", page_icon="😴", layout="wide")

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
            st.error("❌ ไม่พบไฟล์ Sleep_Efficiency.csv")
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
    st.error(f"❌ Error loading/training model: {str(e)}")
    st.stop()

# Header
st.title("😴 Sleep Efficiency Predictor")
st.markdown("ทำนายประสิทธิภาพการนอนหลับด้วย **Support Vector Machine**")
st.markdown("---")

# Sidebar - Input Form
st.sidebar.header("📝 กรอกข้อมูลการนอน")

with st.sidebar.form("input_form"):
    age = st.number_input("Age (ปี)", min_value=1, max_value=120, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    bedtime_hour = st.slider("Bedtime Hour (0-23)", 0, 23, 23)
    bedtime_min = st.slider("Bedtime Minute", 0, 59, 0)
    wakeup_hour = st.slider("Wakeup Hour (0-23)", 0, 23, 7)
    wakeup_min = st.slider("Wakeup Minute", 0, 59, 0)
    
    rem = st.slider("REM Sleep %", 0, 100, 20)
    deep = st.slider("Deep Sleep %", 0, 100, 50)
    light = st.slider("Light Sleep %", 0, 100, 30)
    
    awakenings = st.number_input("Awakenings (ครั้ง)", 0, 20, 1)
    caffeine = st.number_input("Caffeine (mg)", 0, 500, 50)
    alcohol = st.number_input("Alcohol (drinks)", 0, 20, 0)
    smoking = st.selectbox("Smoking Status", ["Yes", "No"])
    exercise = st.slider("Exercise Frequency (days/week)", 0, 7, 3)
    
    submitted = st.form_submit_button("🔮 ทำนาย", use_container_width=True)

# Calculate features
def calculate_features():
    bedtime_total = bedtime_hour + bedtime_min / 60
    wakeup_total = wakeup_hour + wakeup_min / 60
    
    duration = wakeup_total - bedtime_total
    if duration < 0:
        duration += 24
    
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
        'Gender': [gender],
        'Smoking status': [smoking]
    }
    return pd.DataFrame(data)

# Predict
if submitted:
    input_df = calculate_features()
    prediction = model.predict(input_df)[0]
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Sleep Efficiency", f"{prediction:.2%}")
    
    with col2:
        if prediction >= 0.85:
            level = "🟢 ดีมาก"
        elif prediction >= 0.70:
            level = "🟡 ปานกลาง"
        else:
            level = "🔴 ควรปรับปรุง"
        st.metric("📊 ระดับ", level)
    
    with col3:
        st.metric("⏱️ Duration", f"{input_df['Sleep_duration_calc'][0]:.1f} hrs")
    
    # Progress bar
    st.progress(min(max(prediction, 0), 1))
    
    # Recommendations
    st.markdown("### 💡 คำแนะนำ")
    if prediction < 0.70:
        st.warning("""
        - ลดคาเฟอีนและแอลกอฮอล์ก่อนนอน
        - เข้านอนและตื่นนอนเวลาเดิมเป็นประจำ
        - เพิ่มการออกกำลังกายสม่ำเสมอ
        - ลดจำนวนครั้งการตื่นกลางคืน
        """)
    elif prediction < 0.85:
        st.info("""
        - พยายามรักษาเวลาเข้านอนให้สม่ำเสมอ
        - สร้างสภาพแวดล้อมในห้องนอนที่เหมาะสม
        """)
    else:
        st.success("🎉 การนอนหลับของคุณมีคุณภาพดีมาก! รักษาไว้ครับ")