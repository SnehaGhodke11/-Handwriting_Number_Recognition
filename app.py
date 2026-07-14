import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# 🎨 App setup
st.set_page_config(page_title="Handwriting Number Project", page_icon="✍️", layout="centered")
st.title("✍️ Handwriting Number Recognition")
st.markdown("### 🔢 Predict handwritten digits using a trained neural network")

# 🧠 Load trained model (.keras file)
model = load_model("digit_recognition_model.keras")

# 📥 Load MNIST test data
(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test / 255.0  # normalize

# 🔢 User input digit index
index = st.number_input("Enter test sample index (0–9999)", min_value=0, max_value=9999, step=1)

if st.button("Predict"):
    img = X_test[index].reshape(1, 784)
    y_predict = model.predict(img)
    predicted_digit = np.argmax(y_predict[0])

    st.subheader("🧩 Pixel Form of Selected Digit")
    fig, ax = plt.subplots()
    ax.matshow(X_test[index], cmap="gray")
    st.pyplot(fig)

    st.success(f"✅ Predicted Digit: **{predicted_digit}**")
    st.write(f"Confidence: {np.max(y_predict[0])*100:.2f}%")

# 🧭 Sidebar info
st.sidebar.header("About Project")
st.sidebar.info("""
This project uses a deep neural network with dropout layers to reduce overfitting.
Model trained on MNIST dataset for 5 epochs using Adam optimizer.
Accuracy achieved: ~97%.
""")

st.sidebar.markdown("👩‍💻 Created by **Sneha Ghodke**")
st.sidebar.markdown("📚 Dataset: MNIST Handwritten Digits")
st.sidebar.markdown("⚙️ Framework: TensorFlow + Streamlit")
st.sidebar.markdown("💡 Tip: Try different sample indexes to see various digits!")
