@st.cache_resource
def load_model():
    return tf.keras.models.load_model('signature_expert_model.keras', compile=False)

model = load_model()
