import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report
)

# ====== ตั้งค่าหน้าเว็บ ======
st.set_page_config(
    page_title="ML Prediction Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== CSS สไตล์หรูหรา ======
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');

* { font-family: 'Kanit', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
    background-attachment: fixed;
}

/* Header Styles */
.main-header {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 1.5rem;
    border-radius: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    color: white;
    position: relative;
    overflow: hidden;
}

.header-sleep {
    background: linear-gradient(120deg, #4e73df 0%, #224abe 100%);
}

.header-heart {
    background: linear-gradient(120deg, #e74a3b 0%, #c83a30 100%);
}

.header-home {
    background: linear-gradient(120deg, #1cc88a 0%, #16a370 100%);
}

.header-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header-subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
}

/* Card Styles */
.input-card, .result-card, .metric-card, .recommendation-card {
    background: white;
    border-radius: 1rem;
    padding: 1.5rem;
    box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    margin-bottom: 1rem;
    border: 1px solid #f0f2f6;
    transition: transform 0.3s ease;
}

.input-card:hover, .result-card:hover {
    transform: translateY(-3px);
}

.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
}

.section-title-sleep { color: #4e73df; }
.section-title-heart { color: #e74a3b; }

.section-title::before {
    content: "";
    display: inline-block;
    width: 6px;
    height: 20px;
    border-radius: 3px;
    margin-right: 10px;
}

.section-title-sleep::before { background: #4e73df; }
.section-title-heart::before { background: #e74a3b; }

/* Button Styles */
.stButton > button {
    width: 100%;
    height: 50px;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 1rem;
    color: white;
    border: none;
    transition: all 0.3s ease;
}

.btn-sleep {
    background: linear-gradient(120deg, #4e73df 0%, #224abe 100%);
    box-shadow: 0 4px 10px rgba(78, 115, 223, 0.3);
}

.btn-heart {
    background: linear-gradient(120deg, #e74a3b 0%, #c83a30 100%);
    box-shadow: 0 4px 10px rgba(231, 74, 59, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
}

/* Level Badges */
.level-badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 2rem;
    font-weight: 600;
    font-size: 1.2rem;
    color: white;
}

.level-excellent { background: linear-gradient(120deg, #1cc88a 0%, #16a370 100%); }
.level-moderate { background: linear-gradient(120deg, #f6c23e 0%, #d3a833 100%); color: #4a4a4a; }
.level-poor { background: linear-gradient(120deg, #e74a3b 0%, #c83a30 100%); }

/* Progress Bar */
.progress-container {
    background: #e9ecef;
    border-radius: 1rem;
    height: 25px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-bar {
    height: 100%;
    border-radius: 1rem;
    transition: width 1s ease;
}

/* Sidebar */
.sidebar .sidebar-content {
    background: white;
    border-radius: 1rem;
    padding: 1rem;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1.5rem 0;
    margin-top: 2rem;
    color: #858796;
    font-size: 0.9rem;
    border-top: 1px solid #eaeaea;
}

/* Home Page Cards */
.home-card {
    background: white;
    border-radius: 1.5rem;
    padding: 2rem;
    box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 2px solid transparent;
}

.home-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(58, 59, 69, 0.2);
}

.home-card-sleep:hover { border-color: #4e73df; }
.home-card-heart:hover { border-color: #e74a3b; }

.home-icon { font-size: 4rem; margin-bottom: 1rem; }
.home-title { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem; }
.home-desc { color: #858796; font-size: 1rem; }

/* Metric */
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin: 0.5rem 0;
}

.metric-value-sleep { color: #4e73df; }
.metric-value-heart { color: #e74a3b; }
</style>
""", unsafe_allow_html=True)

# ====== ฟังก์ชันโหลดข้อมูล ======
@st.cache_data
def load_sleep_data():
    df = pd.read_csv('Sleep_Efficiency.csv')
    return df

@st.cache_data
def load_heart_data():
    df = pd.read_csv('Heart3.csv')
    return df

# ====== Train Sleep Model (SVM) ======
@st.cache_resource
def train_sleep_model():
    df = load_sleep_data()
    
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
    
    numeric_features = ['Age', 'REM sleep percentage', 'Deep sleep percentage',
                        'Light sleep percentage', 'Awakenings', 'Caffeine consumption',
                        'Alcohol consumption', 'Exercise frequency',
                        'Bedtime_hour', 'Wakeup_hour', 'Sleep_duration_calc']
    categorical_features = ['Gender', 'Smoking status']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    svm_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('svr', SVR(C=10, epsilon=0.01, kernel='rbf', gamma='scale'))
    ])
    
    svm_pipeline.fit(X_train, y_train)
    
    y_pred = svm_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return svm_pipeline, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# ====== Train Heart Model (Decision Tree) ======
@st.cache_resource
def train_heart_model():
    df = load_heart_data()
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X.columns, {'Accuracy': accuracy}

# ====== โหลดโมเดล ======
try:
    sleep_model, sleep_metrics = train_sleep_model()
    heart_model, heart_features, heart_metrics = train_heart_model()
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
    st.stop()

# ====== Sidebar เมนู ======
st.sidebar.markdown("### 🧠 ML Prediction Hub")
page = st.sidebar.radio(
    "เลือกแอปพลิเคชัน",
    ["🏠 หน้าแรก", " ทำนายการนอนหลับ (SVM)", "❤️ ทำนายโรคหัวใจ (Decision Tree)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 ข้อมูลโมเดล")

if page == "😴 ทำนายการนอนหลับ (SVM)":
    st.sidebar.markdown(f"""
    **โมเดล:** Support Vector Regression<br>
    **MAE:** {sleep_metrics['MAE']:.4f}<br>
    **RMSE:** {sleep_metrics['RMSE']:.4f}<br>
    **R²:** {sleep_metrics['R2']:.4f}
    """, unsafe_allow_html=True)
elif page == "❤️ ทำนายโรคหัวใจ (Decision Tree)":
    st.sidebar.markdown(f"""
    **โมเดล:** Decision Tree Classifier<br>
    **ความแม่นยำ:** {heart_metrics['Accuracy']*100:.2f}%<br>
    **จำนวน Features:** {len(heart_features)}
    """, unsafe_allow_html=True)

# ====== หน้าที่ 1: หน้าแรก ======
if page == "🏠 หน้าแรก":
    st.markdown("""
    <div class="main-header header-home">
        <h1 class="header-title">🧠 ML Prediction Hub</h1>
        <p class="header-subtitle">ศูนย์รวมโมเดล Machine Learning สำหรับการทำนายสุขภาพ</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="home-card home-card-sleep">
            <div class="home-icon">😴</div>
            <div class="home-title">ทำนายประสิทธิภาพการนอนหลับ</div>
            <div class="home-desc">ใช้โมเดล SVM วิเคราะห์คุณภาพการนอนของคุณ</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 เริ่มทำนายการนอนหลับ", key="btn_sleep_home"):
            st.sidebar.markdown("เลือก '😴 ทำนายการนอนหลับ' จากเมนูด้านซ้าย")
    
    with col2:
        st.markdown("""
        <div class="home-card home-card-heart">
            <div class="home-icon">❤️</div>
            <div class="home-title">ทำนายความเสี่ยงโรคหัวใจ</div>
            <div class="home-desc">ใช้ Decision Tree วิเคราะห์ความเสี่ยงโรคหัวใจ</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 เริ่มทำนายโรคหัวใจ", key="btn_heart_home"):
            st.sidebar.markdown("เลือก '❤️ ทำนายโรคหัวใจ' จากเมนูด้านซ้าย")
    
    st.markdown("---")
    st.markdown("### 📈 เปรียบเทียบโมเดล")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #4e73df;">😴 Sleep SVM Model</h3>
            <p><strong>ประเภท:</strong> Regression</p>
            <p><strong>MAE:</strong> {sleep_metrics['MAE']:.4f}</p>
            <p><strong>R² Score:</strong> {sleep_metrics['R2']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #e74a3b;">❤️ Heart Decision Tree</h3>
            <p><strong>ประเภท:</strong> Classification</p>
            <p><strong>Accuracy:</strong> {heart_metrics['Accuracy']*100:.2f}%</p>
            <p><strong>Features:</strong> {len(heart_features)} ตัวแปร</p>
        </div>
        """, unsafe_allow_html=True)

# ====== หน้าที่ 2: ทำนายการนอนหลับ ======
elif page == "😴 ทำนายการนอนหลับ (SVM)":
    st.markdown("""
    <div class="main-header header-sleep">
        <h1 class="header-title">😴 ระบบทำนายประสิทธิภาพการนอนหลับ</h1>
        <p class="header-subtitle">วิเคราะห์คุณภาพการนอนด้วย Support Vector Machine</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title section-title-sleep">ข้อมูลส่วนบุคคล</h3>', unsafe_allow_html=True)
        age = st.number_input("อายุ (ปี)", 1, 120, 35)
        gender = st.selectbox("เพศ", ["Male", "Female"])
        smoking = st.selectbox("สูบบุหรี่", ["Yes", "No"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title section-title-sleep">เวลาการนอน</h3>', unsafe_allow_html=True)
        bedtime_hour = st.slider("ชั่วโมงเข้านอน", 0, 23, 23)
        bedtime_min = st.slider("นาทีเข้านอน", 0, 59, 0)
        wakeup_hour = st.slider("ชั่วโมงตื่นนอน", 0, 23, 7)
        wakeup_min = st.slider("นาทีตื่นนอน", 0, 59, 0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title section-title-sleep">ระยะการนอน</h3>', unsafe_allow_html=True)
        rem = st.slider("REM Sleep %", 0, 100, 20)
        deep = st.slider("Deep Sleep %", 0, 100, 50)
        light = st.slider("Light Sleep %", 0, 100, 30)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title section-title-sleep">พฤติกรรม</h3>', unsafe_allow_html=True)
        awakenings = st.number_input("ตื่นกลางคืน (ครั้ง)", 0, 20, 1)
        caffeine = st.number_input("คาเฟอีน (มก.)", 0, 500, 50)
        alcohol = st.number_input("แอลกอฮอล์ (แก้ว)", 0, 20, 0)
        exercise = st.slider("ออกกำลังกาย (วัน/สัปดาห์)", 0, 7, 3)
        st.markdown('</div>', unsafe_allow_html=True)
    
    predict_sleep = st.button("🔮 ทำนายประสิทธิภาพการนอน", key="btn_sleep")
    
    if predict_sleep:
        bedtime_total = bedtime_hour + bedtime_min / 60
        wakeup_total = wakeup_hour + wakeup_min / 60
        duration = wakeup_total - bedtime_total
        if duration < 0:
            duration += 24
        
        input_data = pd.DataFrame({
            'Age': [age], 'REM sleep percentage': [rem],
            'Deep sleep percentage': [deep], 'Light sleep percentage': [light],
            'Awakenings': [awakenings], 'Caffeine consumption': [caffeine],
            'Alcohol consumption': [alcohol], 'Exercise frequency': [exercise],
            'Bedtime_hour': [bedtime_total], 'Wakeup_hour': [wakeup_total],
            'Sleep_duration_calc': [duration],
            'Gender': [gender], 'Smoking status': [smoking]
        })
        
        prediction = sleep_model.predict(input_data)[0]
        
        st.markdown("---")
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-value metric-value-sleep">{prediction:.2%}</div>
            <p style="text-align:center;">ประสิทธิภาพการนอน</p>
            """, unsafe_allow_html=True)
        with col2:
            if prediction >= 0.85:
                level = "ดีมาก"
                level_class = "level-excellent"
            elif prediction >= 0.70:
                level = "ปานกลาง"
                level_class = "level-moderate"
            else:
                level = "ควรปรับปรุง"
                level_class = "level-poor"
            st.markdown(f'<div style="text-align:center;"><span class="level-badge {level_class}">{level}</span></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-value metric-value-sleep">{duration:.1f}</div>
            <p style="text-align:center;">ชั่วโมงการนอน</p>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="progress-container">
            <div class="progress-bar" style="width: {prediction*100}%; background: linear-gradient(90deg, #1cc88a 0%, #4e73df 100%);"></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ====== หน้าที่ 3: ทำนายโรคหัวใจ ======
elif page == "❤️ ทำนายโรคหัวใจ (Decision Tree)":
    st.markdown("""
    <div class="main-header header-heart">
        <h1 class="header-title">❤️ ระบบทำนายความเสี่ยงโรคหัวใจ</h1>
        <p class="header-subtitle">วิเคราะห์ความเสี่ยงด้วย Decision Tree</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title section-title-heart">ข้อมูลพื้นฐาน</h3>', unsafe_allow_html=True)
        age = st.number_input("อายุ", 20, 100, 50)
        sex = st.selectbox("เพศ", [0, 1], format_func=lambda x: "หญิง" if x == 0 else "ชาย")
        cp = st.selectbox("ประเภทอาการเจ็บหน้าอก", [1, 2, 3, 4], format_func=lambda x: f"Type {x}")
        trestbps = st.number_input("ความดันโลหิต", 80, 200, 130)
        chol = st.number_input("คอเลสเตอรอล", 100, 600, 250)
        fbs = st.selectbox("น้ำตาลในเลือด > 120", [0, 1], format_func=lambda x: "ไม่" if x == 0 else "ใช่")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title section-title-heart">ผลการตรวจ</h3>', unsafe_allow_html=True)
        restecg = st.selectbox("ผล ECG", [0, 1, 2], format_func=lambda x: f"Level {x}")
        thalach = st.number_input("อัตราการเต้นหัวใจสูงสุด", 60, 220, 150)
        exang = st.selectbox("เจ็บหน้าอกเมื่อออกกำลังกาย", [0, 1], format_func=lambda x: "ไม่" if x == 0 else "ใช่")
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1)
        slope = st.selectbox("ความชัน ST Segment", [1, 2, 3], format_func=lambda x: f"Level {x}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    predict_heart = st.button("🔍 วิเคราะห์ความเสี่ยง", key="btn_heart")
    
    if predict_heart:
        input_data = pd.DataFrame({
            'Age': [age], 'Sex': [sex], 'ChestPainType': [cp],
            'RestingBP': [trestbps], 'Cholesterol': [chol], 'FastingBS': [fbs],
            'RestingECG': [restecg], 'MaxHR': [thalach], 'ExerciseAngina': [exang],
            'Oldpeak': [oldpeak], 'ST_Slope': [slope]
        })
        
        prediction = heart_model.predict(input_data)[0]
        probabilities = heart_model.predict_proba(input_data)[0]
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            if prediction == 1:
                st.markdown(f"""
                <h2 style="color: #e74a3b; text-align: center;">⚠️ มีความเสี่ยง</h2>
                <div class="metric-value metric-value-heart">{probabilities[1]*100:.1f}%</div>
                <p style="text-align: center;">โอกาสเป็นโรคหัวใจ</p>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <h2 style="color: #1cc88a; text-align: center;">✅ ความเสี่ยงต่ำ</h2>
                <div class="metric-value" style="color: #1cc88a; text-align: center;">{probabilities[1]*100:.1f}%</div>
                <p style="text-align: center;">โอกาสเป็นโรคหัวใจ</p>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-title section-title-heart">Feature Importance</h3>', unsafe_allow_html=True)
            
            importances = heart_model.feature_importances_
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=importances, y=heart_features, palette='viridis', ax=ax)
            ax.set_title('ความสำคัญของแต่ละปัจจัย', fontsize=14, fontweight='bold')
            ax.set_xlabel('Importance Score')
            st.pyplot(fig)
            plt.close()
            st.markdown('</div>', unsafe_allow_html=True)

# ====== Footer ======
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>พัฒนาด้วย ❤️ โดย Streamlit + Machine Learning</p>
    <p>© 2026 ML Prediction Hub | ข้อมูลนี้ใช้เพื่อการศึกษาเท่านั้น</p>
</div>
""", unsafe_allow_html=True)