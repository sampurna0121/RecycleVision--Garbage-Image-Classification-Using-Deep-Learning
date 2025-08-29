# RecycleVision--Garbage-Image-Classification-Using-Deep-Learning
🚮 RecycleVision is a deep learning project that classifies waste images into categories like plastic, glass, paper, cardboard, metal, and organic waste.
It uses Transfer Learning (MobileNetV2/EfficientNetB0) and is deployed via a Streamlit web app.

📌 Features

Upload any waste image → Get predicted category

Shows confidence score (+ top-3 predictions coming soon)

Model evaluation with confusion matrix & metrics

User-friendly Streamlit interface

🗂 Dataset

Source: Garbage Classification Dataset (Kaggle)

Classes: Cardboard, Glass, Metal, Paper, Plastic, Trash/Organic

Preprocessing: Resized (224x224), normalized, augmented

⚙️ Tech Stack

Python 🐍

TensorFlow / Keras 🤖

Streamlit 🎨

NumPy, Pandas, Matplotlib, Seaborn

📊 Results

Best Model: MobileNetV2

Accuracy: ~88%

Metrics: Balanced Precision, Recall, and F1-Score

Confusion Matrix: Shows clear class separation

🚀 Run Locally
# Clone repo
git clone https://github.com/sampurna0121/RecycleVision.git
cd RecycleVision

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run garbage.py


📌 Future Scope

Add more waste categories

Optimize for IoT / Edge devices

Multilingual support in Streamlit

👩‍💻 Author

Sampurna Sharma
LinkedIn
