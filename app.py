
# app.py

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt


@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()


st.title("Iris Flower Species Predictor")


st.header("Enter flower measurements:")

sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.2)

input_array = np.array([[sepal_length, sepal_width, petal_length, petal_width]])


if st.button("Predict"):
    pred_class = model.predict(input_array)[0]
    st.success(f"**Predicted species:** {pred_class}")

    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_array)[0]
        class_names = model.classes_

        st.subheader("Prediction Probabilities")
        fig, ax = plt.subplots()
        ax.bar(class_names, proba, color='skyblue')
        ax.set_ylabel("Probability")
        ax.set_xlabel("Class")
        st.pyplot(fig)


if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importances")
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    importances = model.feature_importances_
    
    fig2, ax2 = plt.subplots()
    ax2.barh(feature_names, importances, color='lightgreen')
    ax2.set_xlabel("Importance Score")
    st.pyplot(fig2)


st.markdown("Please fill in the details below to proceed.")



