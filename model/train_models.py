"""
ML Assignment 2 - Model Training Script
Bank Marketing Dataset - Term Deposit Subscription Prediction
Implements 6 classification models with comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class BankMarketingMLPipeline:
    """
    Complete ML pipeline for Bank Marketing dataset classification
    Predicts whether a client will subscribe to a term deposit
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the pipeline with dataset path
        
        Args:
            dataset_path (str): Path to the bank.csv dataset
        """
        self.dataset_path = dataset_path
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self, target_column='deposit', test_size=0.2, random_state=42):
        """
        Load Bank Marketing dataset and perform preprocessing
        
        Args:
            target_column (str): Name of the target column (default: 'deposit')
            test_size (float): Proportion of test data
            random_state (int): Random seed for reproducibility
        """
        print("="*70)
        print("LOADING BANK MARKETING DATASET")
        print("="*70)
        
        # Load dataset
        df = pd.read_csv(self.dataset_path)
        
        print(f"\n Dataset loaded successfully!")
        print(f"  - Total samples: {df.shape[0]}")
        print(f"  - Total features: {df.shape[1] - 1}")
        print(f"  - Target variable: {target_column}")
        
        # Display target distribution
        print(f"\nTarget Distribution:")
        target_counts = df[target_column].value_counts()
        for label, count in target_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  - {label}: {count} ({percentage:.2f}%)")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print(f"\nPreprocessing Data...")
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        print(f"  - Encoding {len(categorical_cols)} categorical features")
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Encode target variable
        y = self.target_encoder.fit_transform(y)
        print(f"  - Target encoded: {dict(zip(self.target_encoder.classes_, self.target_encoder.transform(self.target_encoder.classes_)))}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\n Data preprocessing completed!")
        print(f"  - Training samples: {self.X_train.shape[0]}")
        print(f"  - Test samples: {self.X_test.shape[0]}")
        print(f"  - Number of features: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_all_models(self):
        """
        Train all 6 required classification models
        """
        print("\n" + "="*70)
        print("TRAINING CLASSIFICATION MODELS")
        print("="*70)
        
        # 1. Logistic Regression
        print("\n[1/6] Training Logistic Regression...")
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, random_state=42, solver='liblinear'
        )
        self.models['Logistic Regression'].fit(self.X_train, self.y_train)
        print("Logistic Regression trained successfully")
        
        # 2. Decision Tree
        print("\n[2/6] Training Decision Tree Classifier...")
        self.models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=10, min_samples_split=20, random_state=42
        )
        self.models['Decision Tree'].fit(self.X_train, self.y_train)
        print("Decision Tree trained successfully")
        
        # 3. K-Nearest Neighbors
        print("\n[3/6] Training K-Nearest Neighbors...")
        self.models['kNN'] = KNeighborsClassifier(n_neighbors=7, weights='distance')
        self.models['kNN'].fit(self.X_train, self.y_train)
        print("kNN trained successfully")
        
        # 4. Naive Bayes
        print("\n[4/6] Training Naive Bayes (Gaussian)...")
        self.models['Naive Bayes'] = GaussianNB()
        self.models['Naive Bayes'].fit(self.X_train, self.y_train)
        print("Naive Bayes trained successfully")
        
        # 5. Random Forest (Ensemble)
        print("\n[5/6] Training Random Forest (Ensemble)...")
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=20, random_state=42
        )
        self.models['Random Forest'].fit(self.X_train, self.y_train)
        print("Random Forest trained successfully")
        
        # 6. XGBoost (Ensemble)
        print("\n[6/6] Training XGBoost (Ensemble)...")
        self.models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            random_state=42, eval_metric='logloss', use_label_encoder=False
        )
        self.models['XGBoost'].fit(self.X_train, self.y_train)
        print("XGBoost trained successfully")
        
        print("\n" + "="*70)
        print(" ALL 6 MODELS TRAINED SUCCESSFULLY!")
        print("="*70)
        
    def evaluate_model(self, model_name, model):
        """
        Evaluate a single model with all required metrics
        
        Args:
            model_name (str): Name of the model
            model: Trained model object
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(self.X_test)
        
        # For AUC score, we need probabilities
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)
                # Binary classification
                if len(np.unique(self.y_test)) == 2:
                    auc_score = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                else:
                    auc_score = roc_auc_score(
                        self.y_test, y_pred_proba, 
                        multi_class='ovr', average='weighted'
                    )
            else:
                auc_score = 0.0
        except:
            auc_score = 0.0
        
        # Calculate all required metrics
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'AUC': auc_score,
            'Precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'F1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
            'MCC': matthews_corrcoef(self.y_test, y_pred)
        }
        
        # Store confusion matrix and classification report
        metrics['confusion_matrix'] = confusion_matrix(self.y_test, y_pred)
        metrics['classification_report'] = classification_report(
            self.y_test, y_pred, 
            target_names=self.target_encoder.classes_
        )
        
        return metrics
    
    def evaluate_all_models(self):
        """
        Evaluate all trained models and store results
        """
        print("\n" + "="*70)
        print("EVALUATING MODELS")
        print("="*70)
        
        for idx, (model_name, model) in enumerate(self.models.items(), 1):
            print(f"\n[{idx}/6] Evaluating {model_name}...")
            self.results[model_name] = self.evaluate_model(model_name, model)
            print(f"{model_name} evaluated successfully")
        
        print("\n" + "="*70)
        print("ALL MODELS EVALUATED SUCCESSFULLY!")
        print("="*70)
        
    def save_models(self, save_dir='.'):
        """
        Save all trained models to disk
        
        Args:
            save_dir (str): Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        for idx, (model_name, model) in enumerate(self.models.items(), 1):
            filename = f"{save_dir}/{model_name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, filename)
            print(f"[{idx}/6] Saved {model_name}")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, f"{save_dir}/scaler.pkl")
        joblib.dump(self.target_encoder, f"{save_dir}/target_encoder.pkl")
        joblib.dump(self.label_encoders, f"{save_dir}/label_encoders.pkl")
        
        print("\n All models and preprocessing objects saved!")
        
    def generate_comparison_table(self):
        """
        Generate comparison table of all models (for README)
        
        Returns:
            pd.DataFrame: Comparison table with all metrics
        """
        comparison_data = []
        
        model_order = [
            'Logistic Regression',
            'Decision Tree',
            'kNN',
            'Naive Bayes',
            'Random Forest',
            'XGBoost'
        ]
        
        for model_name in model_order:
            if model_name in self.results:
                metrics = self.results[model_name]
                comparison_data.append({
                    'ML Model Name': model_name,
                    'Accuracy': f"{metrics['Accuracy']:.4f}",
                    'AUC': f"{metrics['AUC']:.4f}",
                    'Precision': f"{metrics['Precision']:.4f}",
                    'Recall': f"{metrics['Recall']:.4f}",
                    'F1': f"{metrics['F1']:.4f}",
                    'MCC': f"{metrics['MCC']:.4f}"
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        return df_comparison
    
    def display_results(self):
        """
        Display comprehensive results for all models
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON RESULTS")
        print("="*70 + "\n")
        
        df_comparison = self.generate_comparison_table()
        print(df_comparison.to_string(index=False))
        
        print("\n" + "="*70)
        print("DETAILED METRICS FOR EACH MODEL")
        print("="*70)
        
        for model_name, metrics in self.results.items():
            print(f"\n{'='*70}")
            print(f"{model_name.upper()}")
            print(f"{'='*70}")
            print(f"Accuracy:  {metrics['Accuracy']:.4f}")
            print(f"AUC Score: {metrics['AUC']:.4f}")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"Recall:    {metrics['Recall']:.4f}")
            print(f"F1 Score:  {metrics['F1']:.4f}")
            print(f"MCC Score: {metrics['MCC']:.4f}")
            print(f"\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
            print(f"\nClassification Report:")
            print(metrics['classification_report'])
    
    def generate_readme_table(self):
        """
        Generate markdown table for README.md
        """
        print("\n" + "="*70)
        print("MARKDOWN TABLE FOR README.md")
        print("="*70 + "\n")
        
        print("Copy this table to your README.md:\n")
        print("| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |")
        print("|--------------|----------|-----|-----------|--------|----|----|")
        
        model_order = [
            'Logistic Regression',
            'Decision Tree',
            'kNN',
            'Naive Bayes',
            'Random Forest',
            'XGBoost'
        ]
        
        for model_name in model_order:
            if model_name in self.results:
                metrics = self.results[model_name]
                print(f"| {model_name} | {metrics['Accuracy']:.4f} | {metrics['AUC']:.4f} | "
                      f"{metrics['Precision']:.4f} | {metrics['Recall']:.4f} | "
                      f"{metrics['F1']:.4f} | {metrics['MCC']:.4f} |")


def main():
    """
    Main execution function for Bank Marketing dataset
    """
    print("\n" + "="*70)
    print("ML ASSIGNMENT 2 - BANK MARKETING CLASSIFICATION")
    print("Predicting Term Deposit Subscription")
    print("="*70 + "\n")
    
    # Dataset configuration
    dataset_path = 'bank.csv'
    target_column = 'deposit'
    
    # Initialize pipeline
    print("Initializing ML Pipeline...")
    pipeline = BankMarketingMLPipeline(dataset_path)
    
    # Load and preprocess data
    pipeline.load_and_preprocess_data(target_column=target_column)
    
    # Train all models
    pipeline.train_all_models()
    
    # Evaluate all models
    pipeline.evaluate_all_models()
    
    # Display results
    pipeline.display_results()
    
    # Save models
    pipeline.save_models()
    
    # Save comparison table to CSV
    df_comparison = pipeline.generate_comparison_table()
    df_comparison.to_csv('model_comparison.csv', index=False)
    print(f"\n Comparison table saved to 'model_comparison.csv'")
    
    # Generate README table
    pipeline.generate_readme_table()
    
    print("\n" + "="*70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Copy the comparison table to README.md")
    print("2. Write observations for each model")
    print("3. Test the Streamlit app: streamlit run app.py")
    print("4. Deploy to Streamlit Cloud")
    print("5. Submit on Taxila")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
