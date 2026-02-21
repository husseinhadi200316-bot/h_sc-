import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# 1. إعداد واجهة الموقع
st.set_page_config(page_title="كاشف التواقيع الذكي", page_icon="✍️", layout="wide")
st.title("🔍 نظام فحص صحة التواقيع المزوره والحقيقيه")
st.write("")

# 2. تحميل الموديل
@st.cache_resource
def load_my_model():
    model_path = 'signature_expert_model.keras'
    return tf.keras.models.load_model(model_path, compile=False)

model = load_my_model()

# 3. تفعيل خاصية الرفع المتعدد
uploaded_files = st.file_uploader("ارفع صور التواقيع (JPG/PNG)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.divider()
    # عرض النتائج في شبكة (Grid)
    cols = st.columns(2) # سيتم عرض صورتين في كل صف
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # توزيع الصور على الأعمدة
        with cols[idx % 2]:
            with st.container(border=True):
                img = Image.open(uploaded_file)
                st.image(img, caption=f"صورة: {uploaded_file.name}", use_container_width=True)
                
                # معالجة الصورة للموديل
                img_resized = img.resize((224, 224))
                img_array = image.img_to_array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # التنبؤ
                prediction = model.predict(img_array, verbose=0)
                score = prediction[0][0]
                
                if score > 0.5:
                    st.success(f"**النتيجة: حقيقي ✅**")
                    st.caption(f"نسبة الثقة: {score*100:.1f}%")
                else:
                    st.error(f"**النتيجة: مزيف ❌**")
                    st.caption(f"نسبة الثقة: {(1-score)*100:.1f}%")
