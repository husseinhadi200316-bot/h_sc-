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

# 2. دالة تحميل الموديل برابط مباشر
@st.cache_resource
def download_model():
    # هذا هو الرابط المباشر للملف وليس للمجلد
    url = "https://drive.google.com/uc?export=download&id=1Xl_B0pW4979eP55X4Pq8Yf4_p2R9o-mS" 
    output = "model.keras"
    
    if not os.path.exists(output):
        with st.spinner('انتظر قليلاً.. يتم تحميل الموديل من Google Drive...'):
            try:
                urllib.request.urlretrieve(url, output)
            except Exception as e:
                st.error(f"فشل التحميل: {e}")
                
    # إضافة compile=False لحل مشكلة الـ ValueError التي واجهتك
    return tf.keras.models.load_model(output, compile=False)

# تنفيذ التحميل
try:
    model = download_model()
except Exception as e:
    st.error(f"حدث خطأ أثناء تشغيل الموديل: {e}")
    st.stop()

# 3. واجهة رفع الصور
uploaded_file = st.file_uploader("ارفع صورة التوقيع (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="الصورة المرفوعة", width=300)
    
    # تحضير الصورة للموديل
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # إجراء التنبؤ
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