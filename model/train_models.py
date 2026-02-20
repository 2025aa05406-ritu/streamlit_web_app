
import pandas as pd
import numpy as np
import os
import joblib
import warnings
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

warnings.filterwarnings('ignore')

class BankMarketingTrainer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.save_path = 'model'
        os.makedirs(self.save_path, exist_ok=True)
        
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'kNN': KNeighborsClassifier(n_neighbors=7),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
        }
        self.results = []

    def prepare_data(self):
        df = pd.read_csv(self.filepath)
        # Verify minimum requirements: 12 features, 500 instances [cite: 30]
        print(f"Dataset Shape: {df.shape}")
        
        X = df.drop('deposit', axis=1)
        y = df['deposit']
        
        # Encoding categorical features
        le_dict = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            le_dict[col] = le
        
        target_le = LabelEncoder()
        y = target_le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save preprocessing artifacts 
        joblib.dump(scaler, f"{self.save_path}/scaler.pkl")
        joblib.dump(le_dict, f"{self.save_path}/label_encoders.pkl")
        joblib.dump(target_le, f"{self.save_path}/target_encoder.pkl")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        print("\n" + "="*30)
        print("TRAINING 6 MODELS")
        print("="*30)
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # Calculating all 6 mandatory metrics [cite: 41-46]
            metrics = {
                'ML Model Name': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'AUC': roc_auc_score(y_test, y_proba),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1': f1_score(y_test, y_pred, average='weighted'),
                'MCC': matthews_corrcoef(y_test, y_pred)
            }
            self.results.append(metrics)
            
            # Save the model file 
            joblib.dump(model, f"{self.save_path}/{name.replace(' ', '_').lower()}.pkl")
            print(f"✓ {name} trained and saved.")

    def generate_readme_table(self):
        df_res = pd.DataFrame(self.results)
        print("\n" + "="*50)
        print("COPY THIS TABLE TO YOUR README.md")
        print("="*50)
        print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    trainer = BankMarketingTrainer('bank.csv')
    trainer.train_and_evaluate()
    trainer.generate_readme_table()
