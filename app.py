import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import urllib.request
import os

# 1. إعداد واجهة الموقع
st.set_page_config(page_title="كاشف التواقيع", page_icon="✍️")
st.title("🔍 نظام فحص صحة التوقيع")

# 2. تحميل الموديل برابط مباشر
@st.cache_resource
def download_model():
    # استبدل هذا الرابط بالرابط المباشر الذي جهزناه في الخطوة الأولى
    url = "https://drive.google.com/drive/folders/17R5VsTbAv0OTBt2IogW2frxyaldgacJr?usp=drive_link"
    output = "model.keras"
    if not os.path.exists(output):
        with st.spinner('انتظر قليلاً.. يتم تجهيز الذكاء الاصطناعي...'):
            urllib.request.urlretrieve(url, output)
    return tf.keras.models.load_model(output)

model = download_model()

# 3. مكان رفع الصور
uploaded_file = st.file_uploader("ارفع صورة التوقيع (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="الصورة المرفوعة", width=300)
    
    # تحضير الصورة للموديل
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # النتيجة
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    if score > 0.5:
        st.success(f"النتيجة: توقيع حقيقي ✅ (نسبة الثقة: {score*100:.1f}%)")
    else:
        st.error(f"النتيجة: توقيع مزيف ❌ (نسبة الثقة: {(1-score)*100:.1f}%)")