"""
ML Assignment 2 - Streamlit Web Application
Bank Marketing Dataset - Term Deposit Prediction
Interactive frontend for model demonstration and evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Bank Marketing ML App",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)


class BankMarketingMLApp:
    """
    Streamlit application for Bank Marketing ML model deployment
    """
    
    def __init__(self):
        self.models = {}
        self.model_names = [
            'Logistic Regression',
            'Decision Tree',
            'kNN',
            'Naive Bayes',
            'Random Forest',
            'XGBoost'
        ]
        self.scaler = None
        self.target_encoder = None
        self.label_encoders = None
        
    def load_models(self, model_dir='model'):
        """
        Load all saved models from disk
        """
        try:
            for model_name in self.model_names:
                filename = f"{model_dir}/{model_name.replace(' ', '_').lower()}.pkl"
                if os.path.exists(filename):
                    self.models[model_name] = joblib.load(filename)
            
            # Load preprocessing objects
            if os.path.exists(f"{model_dir}/scaler.pkl"):
                self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
            if os.path.exists(f"{model_dir}/target_encoder.pkl"):
                self.target_encoder = joblib.load(f"{model_dir}/target_encoder.pkl")
            if os.path.exists(f"{model_dir}/label_encoders.pkl"):
                self.label_encoders = joblib.load(f"{model_dir}/label_encoders.pkl")
                
            return len(self.models) > 0
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def preprocess_data(self, df, target_column='deposit'):
        """
        Preprocess uploaded bank marketing data
        """
        # Check if target column exists
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df
            y = None
        
        # Handle categorical features using saved encoders
        X_processed = X.copy()
        
        if self.label_encoders is not None:
            for col in X_processed.columns:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    try:
                        X_processed[col] = le.transform(X_processed[col].astype(str))
                    except:
                        # If new categories, use fit_transform
                        le_new = LabelEncoder()
                        X_processed[col] = le_new.fit_transform(X_processed[col].astype(str))
                elif X_processed[col].dtype == 'object':
                    # New categorical column
                    le_new = LabelEncoder()
                    X_processed[col] = le_new.fit_transform(X_processed[col].astype(str))
        else:
            # No saved encoders, encode all object columns
            for col in X_processed.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.mean())
        
        # Scale features
        if self.scaler is not None:
            try:
                X_processed = self.scaler.transform(X_processed)
            except:
                scaler = StandardScaler()
                X_processed = scaler.fit_transform(X_processed)
        else:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
        
        # Encode target if it exists
        if y is not None:
            if self.target_encoder is not None:
                try:
                    y = self.target_encoder.transform(y)
                except:
                    le = LabelEncoder()
                    y = le.fit_transform(y)
            else:
                le = LabelEncoder()
                y = le.fit_transform(y)
        
        return X_processed, y
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate all evaluation metrics
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'MCC Score': matthews_corrcoef(y_true, y_pred)
        }
        
        # Calculate AUC if probabilities are available
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['AUC Score'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['AUC Score'] = roc_auc_score(
                        y_true, y_pred_proba, 
                        multi_class='ovr', average='weighted'
                    )
            except:
                metrics['AUC Score'] = 0.0
        else:
            metrics['AUC Score'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(self, cm, title="Confusion Matrix"):
        """
        Create confusion matrix heatmap
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use target class names if available
        labels = ['No', 'Yes']
        if self.target_encoder is not None:
            labels = self.target_encoder.classes_
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    square=True, cbar_kws={'label': 'Count'},
                    xticklabels=labels, yticklabels=labels,
                    ax=ax)
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        return fig
    
    def run(self):
        """
        Main application logic
        """
        # Header
        st.title("Bank Marketing ML Classification App")
        st.markdown("### Term Deposit Subscription Prediction")
        st.markdown("**M.Tech (AIML/DSE) - Machine Learning Assignment 2**")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.header("Configuration Panel")
        st.sidebar.markdown("---")
        
        # Load models
        with st.spinner("Loading trained models..."):
            models_loaded = self.load_models()
        
        if not models_loaded or len(self.models) == 0:
            st.sidebar.error("Models not found!")
            st.warning("**Models not found.** Please train the models first using `train_models.py`")
            st.info("""
            **Steps to train models:**
            1. Place `bank.csv` in the same directory as `train_models.py`
            2. Run: `python train_models.py`
            3. Wait for training to complete
            4. Refresh this page
            """)
        else:
            st.sidebar.success(f"{len(self.models)} models loaded!")
        
        # Model selection
        st.sidebar.subheader("Model Selection")
        selected_model = st.sidebar.selectbox(
            "Choose a classification model:",
            options=self.model_names if self.models else ["No models available"],
            help="Select the model you want to use for predictions"
        )
        
        # Model info
        if selected_model in self.models:
            st.sidebar.markdown(f"**Selected:** `{selected_model}`")
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("###About Dataset")
        st.sidebar.info("""
        **Bank Marketing Dataset**
        - **Goal**: Predict term deposit subscription
        - **Features**: 16 input features
        - **Target**: deposit (yes/no)
        - **Source**: UCI ML Repository
        """)
        
        # Main content
        st.subheader("Upload Test Dataset")
        st.markdown("Upload your test data (CSV format) to evaluate the model. Due to free tier limitations, upload test data only.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload bank marketing test dataset"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                
                st.success(f"Dataset loaded successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", df.shape[0])
                with col2:
                    st.metric("Total Columns", df.shape[1])
                with col3:
                    st.metric("Features", df.shape[1] - 1 if 'deposit' in df.columns else df.shape[1])
                
                # Show data preview
                with st.expander("Preview Dataset", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    st.markdown("**Dataset Info:**")
                    buffer = []
                    for col in df.columns:
                        buffer.append(f"- `{col}`: {df[col].dtype}")
                    st.markdown("\n".join(buffer[:8]))
                    if len(buffer) > 8:
                        st.markdown(f"*...and {len(buffer) - 8} more columns*")
                
                # Target column
                st.markdown("---")
                
                if 'deposit' in df.columns:
                    target_column = 'deposit'
                    st.info(f"Target column detected: **{target_column}**")
                else:
                    target_column = st.selectbox(
                        "Select the target column:",
                        options=df.columns.tolist(),
                        help="Choose the column containing true labels"
                    )
                
                # Predict button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    predict_button = st.button("Run Prediction & Evaluation", type="primary", use_container_width=True)
                
                if predict_button:
                    if selected_model not in self.models:
                        st.error("Selected model not available. Please train models first.")
                    else:
                        with st.spinner("Processing data and making predictions..."):
                            # Preprocess data
                            X_test, y_test = self.preprocess_data(df, target_column)
                            
                            if y_test is None:
                                st.error("Target column not found or invalid.")
                            else:
                                # Get selected model
                                model = self.models[selected_model]
                                
                                # Make predictions
                                y_pred = model.predict(X_test)
                                
                                # Get probabilities if available
                                y_pred_proba = None
                                if hasattr(model, 'predict_proba'):
                                    y_pred_proba = model.predict_proba(X_test)
                                
                                # Calculate metrics
                                metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
                                
                                # Display results
                                st.markdown("---")
                                st.success(f"Prediction completed successfully!")
                                st.subheader(f"Results: {selected_model}")
                                
                                # Metrics display
                                st.markdown("####Evaluation Metrics")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                                    st.metric("Precision", f"{metrics['Precision']:.4f}")
                                
                                with col2:
                                    st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
                                    st.metric("Recall", f"{metrics['Recall']:.4f}")
                                
                                with col3:
                                    st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                                    st.metric("MCC Score", f"{metrics['MCC Score']:.4f}")
                                
                                st.markdown("---")
                                
                                # Confusion Matrix and Classification Report in columns
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("####Confusion Matrix")
                                    cm = confusion_matrix(y_test, y_pred)
                                    fig = self.plot_confusion_matrix(
                                        cm, 
                                        title=f"Confusion Matrix"
                                    )
                                    st.pyplot(fig)
                                
                                with col2:
                                    st.markdown("####Classification Report")
                                    
                                    # Get class names
                                    if self.target_encoder is not None:
                                        target_names = self.target_encoder.classes_
                                    else:
                                        target_names = ['No', 'Yes']
                                    
                                    report = classification_report(
                                        y_test, y_pred, 
                                        target_names=target_names
                                    )
                                    st.text(report)
                                
                                # Performance summary
                                st.markdown("---")
                                st.markdown("#### 💡 Performance Summary")
                                
                                accuracy_pct = metrics['Accuracy'] * 100
                                
                                if accuracy_pct >= 85:
                                    perf_emoji = "🌟"
                                    perf_text = "Excellent"
                                    perf_color = "green"
                                elif accuracy_pct >= 75:
                                    perf_emoji = "✅"
                                    perf_text = "Good"
                                    perf_color = "blue"
                                elif accuracy_pct >= 65:
                                    perf_emoji = "⚠️"
                                    perf_text = "Moderate"
                                    perf_color = "orange"
                                else:
                                    perf_emoji = "❌"
                                    perf_text = "Needs Improvement"
                                    perf_color = "red"
                                
                                st.markdown(f"""
                                <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; border-left: 5px solid {perf_color};'>
                                    <h4>{perf_emoji} Model Performance: {perf_text}</h4>
                                    <p><strong>The {selected_model} model achieved {accuracy_pct:.2f}% accuracy on the test set.</strong></p>
                                    <ul>
                                        <li>Correctly predicted: <strong>{int(metrics['Accuracy'] * len(y_test))}</strong> out of <strong>{len(y_test)}</strong> samples</li>
                                        <li>Precision: <strong>{metrics['Precision']:.4f}</strong> - Accuracy of positive predictions</li>
                                        <li>Recall: <strong>{metrics['Recall']:.4f}</strong> - Coverage of actual positives</li>
                                        <li>F1 Score: <strong>{metrics['F1 Score']:.4f}</strong> - Harmonic mean of precision and recall</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Download predictions
                                st.markdown("---")
                                st.markdown("####Download Results")
                                
                                # Create results dataframe
                                results_df = pd.DataFrame({
                                    'True_Label': self.target_encoder.inverse_transform(y_test) if self.target_encoder else y_test,
                                    'Predicted_Label': self.target_encoder.inverse_transform(y_pred) if self.target_encoder else y_pred,
                                    'Correct': y_test == y_pred
                                })
                                
                                # Add probabilities if available
                                if y_pred_proba is not None:
                                    results_df['Probability_No'] = y_pred_proba[:, 0]
                                    results_df['Probability_Yes'] = y_pred_proba[:, 1]
                                
                                csv = results_df.to_csv(index=False)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        label="Download Predictions (CSV)",
                                        data=csv,
                                        file_name=f"{selected_model.replace(' ', '_')}_predictions.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                                
                                with col2:
                                    # Create metrics summary
                                    metrics_df = pd.DataFrame([metrics])
                                    metrics_csv = metrics_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Metrics (CSV)",
                                        data=metrics_csv,
                                        file_name=f"{selected_model.replace(' ', '_')}_metrics.csv",
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                with st.expander("View Error Details"):
                    st.exception(e)
        
        else:
            st.info("Please upload a CSV file to get started")
            
            # Show example data format
            with st.expander("Expected Data Format"):
                st.markdown("""
                **Your CSV file should have the following columns:**
                
                - `age`: Age of the client
                - `job`: Type of job
                - `marital`: Marital status
                - `education`: Education level
                - `default`: Has credit in default?
                - `balance`: Account balance
                - `housing`: Has housing loan?
                - `loan`: Has personal loan?
                - `contact`: Contact communication type
                - `day`: Last contact day
                - `month`: Last contact month
                - `duration`: Last contact duration
                - `campaign`: Number of contacts in this campaign
                - `pdays`: Days since last contact
                - `previous`: Number of contacts before this campaign
                - `poutcome`: Outcome of previous campaign
                - `deposit`: Target variable (yes/no)
                
                **Note:** The `deposit` column should be present for evaluation.
                """)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; padding: 20px;'>
                <p><strong>Bank Marketing ML Classification System</strong></p>
                <p>Built using Streamlit | M.Tech AIML/DSE - ML Assignment 2</p>
                <p>Dataset: UCI Bank Marketing | Models: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def main():
    """
    Main entry point
    """
    app = BankMarketingMLApp()
    app.run()


if __name__ == "__main__":
    main()
