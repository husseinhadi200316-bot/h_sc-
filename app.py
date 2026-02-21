import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# 1. إعداد واجهة الموقع
st.set_page_config(page_title="كاشف التواقيع المطور", page_icon="✍️", layout="wide")

# --- إضافة الفلتر في الشريط الجانبي ---
st.sidebar.header("⚙️ خيارات العرض")
filter_option = st.sidebar.selectbox(
    "تصفية النتائج حسب:",
    ["الكل", "الحقيقي فقط ✅", "المزيف فقط ❌"]
)

st.title("🔍 نظام فحص وتصفية التواقيع")
st.write("ارفع التواقيع واستخدم الفلتر من اليسار لتنظيم النتائج")

# 2. تحميل الموديل
@st.cache_resource
def load_my_model():
    model_path = 'signature_expert_model.keras'
    return tf.keras.models.load_model(model_path, compile=False)

model = load_my_model()

# 3. رفع الصور
uploaded_files = st.file_uploader("ارفع صور التواقيع...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    results = [] # قائمة لتخزين البيانات قبل الفلترة
    
    # إجراء التنبؤ لكل الصور أولاً
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        score = prediction[0][0]
        label = "حقيقي" if score > 0.5 else "مزيف"
        
        results.append({
            "file": uploaded_file,
            "img": img,
            "score": score,
            "label": label,
            "name": uploaded_file.name
        })

    # --- تطبيق الفلتر ---
    filtered_results = []
    if filter_option == "الكل":
        filtered_results = results
    elif filter_option == "الحقيقي فقط ✅":
        filtered_results = [r for r in results if r["label"] == "حقيقي"]
    else:
        filtered_results = [r for r in results if r["label"] == "مزيف"]

    # 4. عرض النتائج المفلترة
    st.divider()
    st.subheader(f"النتائج المعروضة: {len(filtered_results)}")
    
    cols = st.columns(3) # عرض 3 صور في الصف الواحد
    for idx, res in enumerate(filtered_results):
        with cols[idx % 3]:
            with st.container(border=True):
                st.image(res["img"], caption=res["name"], use_container_width=True)
                if res["label"] == "حقيقي":
                    st.success(f"حقيقي ✅ ({res['score']*100:.1f}%)")
                else:
                    st.error(f"مزيف ❌ ({(1-res['score'])*100:.1f}%)")
