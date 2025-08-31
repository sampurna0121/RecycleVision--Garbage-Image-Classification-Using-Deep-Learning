import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

# --- Paths ---
MODEL_PATH = "garbage_classifier.h5"
CLASS_INDICES_PATH = "class_indices.json"
TEST_DIR = "data/test"

# --- Load Model ---
@st.cache_resource
def load_garbage_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    return None

model = load_garbage_model()

# --- Load Class Names ---
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    CLASS_NAMES = list(class_indices.keys())
else:
    CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]  # fallback

# --- Preprocess Image ---
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# --- Predict Function ---
def predict_image(img, model):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    return predictions

# --- Evaluate Model ---
def evaluate_model(model, test_dir, sample_limit=None):
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Optionally limit dataset
    if sample_limit:
        test_gen.samples = min(sample_limit, test_gen.samples)
        test_gen._set_index_array()

    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys(), output_dict=True)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics, cm, report

# --- Streamlit UI ---
st.title("♻️ Garbage Classification App")
menu = ["Home", "Classification", "Evaluation", "About"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.header("Welcome to Garbage Classifier")
    st.write("Upload an image to classify waste into categories like cardboard, glass, metal, paper, plastic, and trash.")

elif choice == "Classification":
    st.header("Classify an Image")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if model is not None:
            predictions = predict_image(img, model)
            pred_idx = np.argmax(predictions)
            confidence = predictions[0][pred_idx] * 100

            st.success(f"Prediction: {CLASS_NAMES[pred_idx]} ({confidence:.2f}%)")

            # Top-3 Predictions with progress bars
            st.subheader("Top-3 Predictions")
            top3_idx = predictions[0].argsort()[-3:][::-1]
            for i in top3_idx:
                st.write(f"{CLASS_NAMES[i]}: {predictions[0][i]*100:.2f}%")
                st.progress(float(predictions[0][i]))
        else:
            st.error("Model not found. Please upload a trained model.")

elif choice == "Evaluation":
    st.header("Evaluate Model Performance")
    if model is not None and os.path.exists(TEST_DIR):
        sample_limit = st.sidebar.slider("Limit test samples", min_value=100, max_value=2000, step=100, value=500)
        if st.button("Run Evaluation"):
            metrics, cm, report = evaluate_model(model, TEST_DIR, sample_limit=sample_limit)

            st.subheader("Evaluation Metrics")
            for key, val in metrics.items():
                st.write(f"{key}: {val:.4f}")

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            st.subheader("Classification Report")
            st.json(report)
    else:
        st.error("Model or test data directory not found.")

elif choice == "About":
    st.header("About")
    st.write("This app classifies garbage images into recyclable categories using a deep learning model built with TensorFlow & Keras.")
