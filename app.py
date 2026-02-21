import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# 1. إعداد واجهة الموقع
st.set_page_config(page_title="كاشف التواقيع", page_icon="✍️")
st.title("🔍 نظام فحص صحة التوقيع")

# 2. تحميل الموديل من الملف المرفوع في GitHub
@st.cache_resource
def load_my_model():
    # اسم الملف كما هو موجود في مستودع GitHub الخاص بك
    model_path = 'signature_expert_model.keras'
    
    if not os.path.exists(model_path):
        st.error(f"لم يتم العثور على ملف الموديل باسم {model_path} في GitHub. يرجى رفعه بجانب هذا الملف.")
        st.stop()
        
    # تحميل الموديل مع إيقاف الـ compile لتجنب مشاكل الإصدارات
    return tf.keras.models.load_model(model_path, compile=False)

# محاولة تشغيل الموديل
try:
    model = load_my_model()
except Exception as e:
    st.error(f"حدث خطأ أثناء تحميل الموديل: {e}")
    st.stop()

# 3. واجهة رفع الصور
uploaded_file = st.file_uploader("ارفع صورة التوقيع (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="الصورة المرفوعة", width=300)
    
    # معالجة الصورة لتناسب الموديل
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # التنبؤ
    with st.spinner('جاري التحليل...'):
        prediction = model.predict(img_array)
        score = prediction[0][0]
    
    st.divider()
    if score > 0.5:
        st.success(f"### النتيجة: توقيع حقيقي ✅")
        st.write(f"نسبة الثقة: {score*100:.2f}%")
    else:
        st.error(f"### النتيجة: توقيع مزيف ❌")
        st.write(f"نسبة الثقة: {(1-score)*100:.2f}%")
