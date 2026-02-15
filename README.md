# Bank Marketing ML Classification Project
## M.Tech (AIML/DSE) - Machine Learning Assignment 2

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.4+-orange.svg)](https://scikit-learn.org/)

---

## Problem Statement

This project implements a comprehensive machine learning classification pipeline to **predict whether a client will subscribe to a term deposit** based on data from direct marketing campaigns of a Portuguese banking institution. The goal is to compare the performance of six different classification models and deploy an interactive web application for real-time predictions and model evaluation.

**Objective**: Build, evaluate, and deploy multiple classification models to accurately predict term deposit subscription while comparing their performance across multiple evaluation metrics, helping the bank optimize its marketing strategies and improve campaign effectiveness.

**Business Impact**: This predictive model can help banks:
- Identify high-potential customers for targeted marketing
- Reduce marketing costs by focusing on likely subscribers
- Improve campaign conversion rates
- Optimize resource allocation in marketing campaigns

---

## Dataset Description

### Dataset Information
- **Source**: UCI Machine Learning Repository - Bank Marketing Dataset
- **Dataset Name**: Bank Marketing Dataset (Direct Marketing Campaigns)
- **Type**: Binary Classification
- **Number of Features**: 16 input features (exceeds minimum requirement of 12)
- **Number of Instances**: 11,162 samples (exceeds minimum requirement of 500)
- **Target Variable**: `deposit` (yes/no - Will the client subscribe to a term deposit?)
- **Class Distribution**: 
  - **No**: 5,873 samples (52.6%)
  - **Yes**: 5,289 samples (47.4%)
  - **Balance**: Relatively balanced dataset

### Feature Description

The dataset contains information about:
- **Client Data**: Age, job, marital status, education, default status, balance
- **Campaign Data**: Contact type, day, month, duration, number of contacts
- **Previous Campaign Data**: Days since last contact, previous contacts, outcome

| Feature Name | Type | Description |
|--------------|------|-------------|
| `age` | Numerical | Age of the client (years) |
| `job` | Categorical | Type of job (admin., technician, services, management, etc.) |
| `marital` | Categorical | Marital status (married, single, divorced) |
| `education` | Categorical | Education level (primary, secondary, tertiary, unknown) |
| `default` | Categorical | Has credit in default? (yes, no) |
| `balance` | Numerical | Average yearly balance in euros |
| `housing` | Categorical | Has housing loan? (yes, no) |
| `loan` | Categorical | Has personal loan? (yes, no) |
| `contact` | Categorical | Contact communication type (cellular, telephone, unknown) |
| `day` | Numerical | Last contact day of the month |
| `month` | Categorical | Last contact month of year |
| `duration` | Numerical | Last contact duration in seconds |
| `campaign` | Numerical | Number of contacts performed during this campaign |
| `pdays` | Numerical | Number of days since last contact from previous campaign (-1 means not contacted) |
| `previous` | Numerical | Number of contacts performed before this campaign |
| `poutcome` | Categorical | Outcome of the previous marketing campaign (success, failure, unknown, other) |
| **`deposit`** | **Binary** | **Target: Has the client subscribed to a term deposit? (yes, no)** |

### Data Preprocessing Steps

1. **Categorical Encoding**: Applied Label Encoding to all categorical variables (job, marital, education, default, housing, loan, contact, month, poutcome)
2. **Feature Scaling**: Standardized all numerical features using StandardScaler to ensure equal contribution
3. **Train-Test Split**: 80-20 split (8,930 training samples, 2,232 test samples) with stratification to maintain class balance
4. **No Missing Values**: Dataset has no missing values, requiring no imputation
5. **Class Balance**: Dataset is relatively balanced (52.6% vs 47.4%), no special handling needed

### Dataset Statistics
- **Total Samples**: 11,162
- **Training Samples**: 8,929 (80%)
- **Test Samples**: 2,233 (20%)
- **Features after encoding**: 16
- **Target Classes**: 2 (Binary Classification)

---

## Models Used

### Model Comparison Table

**Run `python train_models.py` and copy the results here:**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Decision Tree | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| kNN | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Naive Bayes | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| Random Forest (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |
| XGBoost (Ensemble) | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX |

> **Note**: After training, the script will output a formatted table. Copy it here to replace the X values.

### Model Performance Observations

**Write your observations after training. Here's a template:**

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | [Example: Achieved strong baseline performance with balanced precision and recall. The linear model effectively captured the relationship between client features and subscription likelihood. Fast training time and interpretable coefficients made it ideal for understanding which features (e.g., duration, balance) most strongly predict subscription. The model performed well on this dataset due to relatively linear decision boundaries between classes.] |
| **Decision Tree** | [Example: Showed good performance but with signs of overfitting despite max_depth constraint. The tree structure provided excellent interpretability, clearly showing decision rules (e.g., if duration > X seconds and age > Y, then likely subscriber). However, performance varied across different test samples due to high variance inherent in single decision trees. Feature importance analysis revealed duration and previous campaign outcome as key predictors.] |
| **kNN** | [Example: Moderate performance with computational overhead during prediction phase. The distance-based approach struggled slightly with the mixed numerical and categorical nature of features. Performance was sensitive to the choice of k value (k=7 provided best results). Benefited significantly from feature scaling. Struggled with high-dimensional feature space after one-hot encoding, leading to curse of dimensionality effects.] |
| **Naive Bayes** | [Example: Fast training and prediction with surprisingly competitive results despite strong independence assumptions. The model's assumption that features are independent likely doesn't hold perfectly (e.g., balance and housing loan are related), which limited its maximum performance. However, it handled class probabilities well and provided good baseline predictions. Particularly effective for clients with clear patterns in previous campaign outcomes.] |
| **Random Forest (Ensemble)** | [Example: Achieved the best overall performance among all models with excellent generalization. The ensemble of decision trees effectively reduced overfitting while maintaining interpretability through feature importance scores. Handled feature interactions naturally (e.g., age and job type combined effects). Robust to outliers in balance and duration features. The model successfully identified that call duration and previous campaign success are the strongest predictors of subscription.] |
| **XGBoost (Ensemble)** | [Example: Demonstrated the highest accuracy and F1-score through sophisticated gradient boosting. Excelled at capturing complex non-linear relationships between features. Built-in regularization prevented overfitting effectively. Longer training time compared to simpler models but superior prediction performance justified the computational cost. Feature importance analysis revealed nuanced interactions between campaign timing, client demographics, and previous contact history that other models missed.] |

### Overall Analysis

**Write 2-3 paragraphs after training. Template:**

The ensemble methods (Random Forest and XGBoost) significantly outperformed individual classifiers, demonstrating the power of combining multiple weak learners. Random Forest achieved the best balance between performance and interpretability, making it the recommended model for deployment in production environments where both accuracy and explainability are crucial. XGBoost showed the highest raw performance metrics but at the cost of increased complexity and training time.

Among individual classifiers, Logistic Regression provided the strongest baseline with excellent interpretability, making it valuable for understanding feature relationships. The Decision Tree offered good performance with intuitive decision rules but showed signs of overfitting. kNN's instance-based learning struggled with the high-dimensional encoded feature space, while Naive Bayes provided fast predictions with acceptable but limited accuracy due to feature independence assumptions.

For production deployment in a bank marketing context, Random Forest is recommended as the primary model due to its excellent performance (85%+ accuracy), reasonable training time, and ability to provide feature importance rankings that can guide marketing strategies. Logistic Regression serves as a valuable secondary model for A/B testing and for scenarios where model interpretability is legally required for decision explanation to clients.

**Key Insights:**
- Call duration is the strongest predictor across all models
- Previous campaign outcome significantly impacts subscription likelihood  
- Ensemble methods provide 8-12% accuracy improvement over single classifiers
- Model selection involves trade-offs: accuracy vs. speed vs. interpretability
- The balanced dataset allowed models to perform well without special class weighting

---