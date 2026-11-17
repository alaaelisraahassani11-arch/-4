import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# نماذج التعلم الآلي + الشبكة العصبية من Scikit-Learn
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# إعدادات الصفحة
st.set_page_config(
    page_title="تحليل سلوك المستخدمين - الشبكة العصبية",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 منصة تحليل سلوك المستخدمين والتنبؤ بالعودة (الشبكة العصبية)")
st.markdown("قم برفع ملف البيانات (Excel أو CSV)، حدد المتغيرات، درب الشبكة العصبية أو النماذج الأخرى، واحصل على التنبؤات!")

# --- الخطوة 1: رفع البيانات ---
st.sidebar.header("1. تحميل البيانات")
uploaded_file = st.sidebar.file_uploader("اختر ملف البيانات (Excel أو CSV)", type=["xlsx", "csv"])

@st.cache_data
def load_data(file):
    if file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    elif file.name.endswith('.csv'):
        return pd.read_csv(file)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("تم تحميل البيانات بنجاح!")
    
    with st.expander("🔍 معاينة البيانات"):
        st.dataframe(df.head())
        st.write(f"أبعاد البيانات: **{df.shape[0]}** صف و **{df.shape[1]}** عمود.")

    # --- الخطوة 2: اختيار المتغيرات والهدف ---
    st.sidebar.header("2. إعدادات المتغيرات")
    all_columns = df.columns.tolist()
    
    default_target_idx = all_columns.index('Returned_Within_7_Days') if 'Returned_Within_7_Days' in all_columns else len(all_columns)-1
    target_column = st.sidebar.selectbox("حدد المتغير المستهدف (Target)", all_columns, index=default_target_idx)
    
    default_features = [col for col in ['Session_Duration_Min', 'Pages_Viewed', 'Clicks', 'Search_Count', 'Device_Type', 'Traffic_Source'] if col in all_columns]
    feature_columns = st.sidebar.multiselect("حدد متغيرات المدخلات (Features)", all_columns, default=default_features)

    if feature_columns and target_column:
        data_model = df[feature_columns + [target_column]].dropna().copy()

        # تحويل الهدف (Yes/No) إلى (1/0)
        if data_model[target_column].dtype == 'object':
            data_model[target_column] = data_model[target_column].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true', 'نعم'] else 0)

        X = data_model[feature_columns]
        y = data_model[target_column]

        test_size = st.sidebar.slider("نسبة بيانات الاختبار (Test Size)", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # --- الخطوة 3: اختيار النموذج ---
        st.sidebar.header("3. اختيار النموذج")
        model_type = st.sidebar.selectbox(
            "اختر نوع النموذج", 
            ["Neural Network (MLP)", "Random Forest", "Logistic Regression"]
        )

        # إعدادات إضافية للشبكة العصبية
        if model_type == "Neural Network (MLP)":
            st.sidebar.subheader("⚙️ إعدادات الشبكة العصبية")
            layer1 = st.sidebar.number_input("خلايا الطبقة المخفية الأولى", min_value=10, max_value=200, value=100, step=10)
            layer2 = st.sidebar.number_input("خلايا الطبقة المخفية الثانية", min_value=0, max_value=200, value=50, step=10)
            max_iter = st.sidebar.slider("عدد دورات التدريب (Epochs/Iterations)", 50, 500, 200)

        if st.sidebar.button("🚀 بدء التدريب والاختبار"):
            with st.spinner("جاري معالجة البيانات وتدريب النموذج..."):
                
                numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), numeric_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                    ]
                )

                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)

                # اختيار وتدريب النموذج
                if model_type == "Neural Network (MLP)":
                    hidden_layers = (layer1, layer2) if layer2 > 0 else (layer1,)
                    model = MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation='relu',
                        solver='adam',
                        max_iter=max_iter,
                        random_state=42,
                        early_stopping=True
                    )
                elif model_type == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                elif model_type == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)

                model.fit(X_train_processed, y_train)
                y_pred = model.predict(X_test_processed)

                # حفظ النموذج والمعالج في الجلسة لاستخدامهما في التنبؤ اليدوي
                st.session_state['model'] = model
                st.session_state['preprocessor'] = preprocessor
                st.session_state['features'] = feature_columns
                st.session_state['numeric_features'] = numeric_features
                st.session_state['categorical_features'] = categorical_features
                st.session_state['X_sample'] = X

                st.success(f"تم تدريب نموذج **{model_type}** بنجاح!")
                
                # مقاييس التقييم
                acc = accuracy_score(y_test, y_pred)
                st.metric("🎯 دقة النموذج على بيانات الاختبار (Accuracy)", f"{acc*100:.2f}%")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📋 تقرير التصنيف (Classification Report)")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

                with col2:
                    st.subheader("📉 مصفوفة الارتباك (Confusion Matrix)")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', cbar=True, ax=ax)
                    ax.set_xlabel('Predicted (المتوقع)')
                    ax.set_ylabel('Actual (الفعلي)')
                    st.pyplot(fig)

        # --- الخطوة 4: قسم التنبؤ الفوري لمستخدم جديد ---
        if 'model' in st.session_state:
            st.markdown("---")
            st.header("🔮 تجربة التنبؤ لمستخدم جديد (Prediction)")
            st.markdown("أدخل بيانات جلسة مستخدم جديد ليتنبأ النموذج هل سيعود خلال 7 أيام أم لا:")
            
            input_data = {}
            cols = st.columns(3)
            for idx, col in enumerate(st.session_state['features']):
                with cols[idx % 3]:
                    if col in st.session_state['numeric_features']:
                        default_val = float(st.session_state['X_sample'][col].median())
                        input_data[col] = st.number_input(f"{col}", value=default_val)
                    else:
                        unique_vals = st.session_state['X_sample'][col].unique().tolist()
                        input_data[col] = st.selectbox(f"{col}", unique_vals)
            
            if st.button("تنبؤ النتيجة الآن"):
                input_df = pd.DataFrame([input_data])
                input_processed = st.session_state['preprocessor'].transform(input_df)
                prediction = st.session_state['model'].predict(input_processed)[0]
                prob = st.session_state['model'].predict_proba(input_processed)[0][1]
                
                if prediction == 1:
                    st.success(f"✅ **النتيجة: سيعود المستخدم خلال 7 أيام (Yes)** — بنسبة احتمال: **{prob*100:.1f}%**")
                else:
                    st.error(f"❌ **النتيجة: لن يعود المستخدم خلال 7 أيام (No)** — نسبة احتمال العودة: **{prob*100:.1f}%**")

    else:
        st.info("الرجاء اختيار متغيرات المدخلات (Features) والمتغير المستهدف (Target) من الشريط الجانبي.")
else:
    st.warning("👈 الرجاء رفع ملف البيانات (`M4.xlsx`) من الشريط الجانبي للبدء.")