# Iris-Flower-Species-Predictor
# 🌸 Iris Flower Species Predictor

This is a web-based machine learning app built using **Streamlit** that predicts the species of an Iris flower based on user-provided flower measurements. The model uses a trained classifier (saved as `model.pkl`) to predict the species and display the prediction along with probability scores and feature importances.

### 🔍 App Highlights

- **Input**: Sepal length, sepal width, petal length, and petal width
- **Prediction Output**: Predicted Iris species (Setosa, Versicolor, or Virginica)
- **Visualization**:
  - Prediction probability bar chart
  - Feature importance bar chart (if available from the model)

### 🛠 Built With

- Python
- Streamlit
- scikit-learn
- NumPy
- Matplotlib
- joblib

### 📁 Model Info

- Pre-trained classification model (`model.pkl`)
- Supports `.predict()` and optionally `.predict_proba()` and `.feature_importances_` (like RandomForest)

### 📦 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
