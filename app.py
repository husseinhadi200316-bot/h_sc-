import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# 1. إعداد واجهة الموقع
st.set_page_config(page_title="كاشف التواقيع المتعدد", page_icon="✍️")
st.title("🔍 نظام فحص صحة التواقيع")

# 2. تحميل الموديل
@st.cache_resource
def load_my_model():
    model_path = 'signature_expert_model.keras'
    return tf.keras.models.load_model(model_path, compile=False)

model = load_my_model()

# 3. التعديل الجوهري: إضافة accept_multiple_files=True
uploaded_files = st.file_uploader("ارفع صور التواقيع (JPG/PNG)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"عدد الصور المرفوعة: {len(uploaded_files)}")
    
    # معالجة كل صورة على حدة
    for uploaded_file in uploaded_files:
        # إنشاء صندوق (Expander) لكل صورة للحفاظ على ترتيب الموقع
        with st.expander(f"نتائج فحص: {uploaded_file.name}"):
            col1, col2 = st.columns([1, 2])
            
            img = Image.open(uploaded_file)
            with col1:
                st.image(img, caption="التوقيع المرفوع", use_container_width=True)
            
            # تحضير الصورة للموديل
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # التنبؤ
            prediction = model.predict(img_array, verbose=0)
            score = prediction[0][0]
            
            with col2:
                if score > 0.5:
                    st.success(f"النتيجة: **توقيع حقيقي ✅**")
                    st.write(f"نسبة الثقة: {score*100:.2f}%")
                else:
                    st.error(f"النتيجة: **توقيع مزيف ❌**")
                    st.write(f"نسبة الثقة: {(1-score)*100:.2f}%")
                
                # عرض شريط تقدم يوضح الثقة
                st.progress(float(score) if score > 0.5 else float(1-score))
