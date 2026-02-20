import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Page configuration
st.set_page_config(page_title="Bank Marketing ML App", page_icon="🏦", layout="wide")

class BankMarketingMLApp:
    def __init__(self):
        self.models = {}
        # Exactly 6 models as required by Step 2
        self.model_names = [
            'Logistic Regression', 'Decision Tree', 'kNN', 
            'Naive Bayes', 'Random Forest', 'XGBoost'
        ]
        self.model_dir = 'model'

    def load_artifacts(self):
        """Loads models and preprocessing objects from the 'model/' directory"""
        try:
            for name in self.model_names:
                path = os.path.join(self.model_dir, f"{name.replace(' ', '_').lower()}.pkl")
                if os.path.exists(path):
                    self.models[name] = joblib.load(path)
            
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.target_encoder = joblib.load(os.path.join(self.model_dir, 'target_encoder.pkl'))
            self.label_encoders = joblib.load(os.path.join(self.model_dir, 'label_encoders.pkl'))
            return len(self.models) == 6
        except:
            return False

    def run(self):
        st.title("🏦 Bank Marketing Classification")
        st.markdown("M.Tech (AIML/DSE) - Assignment 2")

        if not self.load_artifacts():
            st.error("Missing models in 'model/' folder. Run train_models.py first!")
            return

        # Sidebar Dropdown
        selected_model_name = st.sidebar.selectbox("Select Model", self.model_names)
        
        # Dataset Upload
        uploaded_file = st.file_uploader("Upload Test CSV", type=['csv'])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Data Preview:", df.head())

            if st.button("Run Evaluation"):
                # Preprocessing
                X = df.drop(columns=['deposit']) if 'deposit' in df.columns else df
                y_true = self.target_encoder.transform(df['deposit'])

                for col, le in self.label_encoders.items():
                    X[col] = le.transform(X[col].astype(str))
                
                X_scaled = self.scaler.transform(X)
                
                # Prediction
                model = self.models[selected_model_name]
                y_pred = model.predict(X_scaled)
                
                # Metrics Display
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
                col2.metric("MCC Score", f"{matthews_corrcoef(y_true, y_pred):.4f}")
                col3.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")

                # Confusion Matrix
                st.subheader(f"Confusion Matrix: {selected_model_name}")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

if __name__ == "__main__":
    app = BankMarketingMLApp()
    app.run()
