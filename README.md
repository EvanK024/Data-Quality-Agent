# Data Quality Agent

## End-to-End Machine Learning System for Automated Data Quality Detection

An end-to-end machine learning project designed to identify potentially low-quality data records before they reach analysts and business users.

The system uses Python and scikit-learn to train and optimize classification models, then deploys the final model through Microsoft Azure Machine Learning as an inference endpoint.

---

## 🎯 Business Problem

Poor-quality data can create additional manual review work for analysts and reduce confidence in downstream analysis.

This project addresses that problem by developing a machine learning system that predicts whether an incoming data record requires additional quality review.

The goal is not to replace human reviewers, but to prioritize records that are more likely to contain quality issues.

---

## 🧠 Machine Learning Approach

This project uses supervised binary classification to predict:

- `0` — No review required
- `1` — Review required

Three machine learning approaches were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

Because only approximately 10.2% of the records required review, recall and F1 score were emphasized rather than relying on accuracy alone.

---

## 📊 Dataset

The dataset contains **50,000 records** and seven predictive attributes:

| Feature | Description |
|---|---|
| `task_type` | Type of task associated with the record |
| `missing_value_count` | Number of missing values |
| `duplicate_flag` | Indicates whether the record is a duplicate |
| `format_error_flag` | Indicates whether formatting issues were detected |
| `response_length` | Length of the response |
| `review_score` | Quality/review score |
| `latency_seconds` | Processing latency |

### Target Variable

`needs_review`

- `0` = No review required
- `1` = Review required

The dataset contains approximately **10.2% positive records**, creating a class-imbalance challenge.

---

## 🔧 Data Preparation

The data preparation workflow included:

- Missing-value handling using median imputation
- One-hot encoding for categorical variables
- Stratified train/test splitting
- Additional validation data for threshold tuning
- Feature preprocessing using scikit-learn pipelines

The final data split consisted of:

- 32,000 training records
- 8,000 validation records
- 10,000 test records

---

## 📈 Model Evaluation

The following models were evaluated during development:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Baseline Logistic Regression | 90.23% | 63.80% | 10.18% | 17.55% |
| Class-Weighted Logistic Regression | 71.99% | 21.63% | 66.34% | 32.62% |
| Threshold-Tuned Logistic Regression | 86.15% | 33.95% | 37.48% | 35.63% |
| Hyperparameter-Tuned Logistic Regression | 86.15% | 33.95% | 37.48% | 35.63% |

### Key Finding

The baseline model achieved high accuracy but detected relatively few records requiring review.

Class weighting substantially improved recall, while threshold tuning provided a better balance between precision and recall.

The final model uses a **0.70 probability threshold** to determine whether a record should be flagged for review.

---

## ⚙️ Model Optimization

Several optimization strategies were explored.

### Class Weighting

Class weighting was used to give greater importance to the minority class.

This increased recall from approximately **10% to 66%**, although it reduced overall accuracy.

### Probability Threshold Tuning

Instead of using the default 0.50 classification threshold, validation data was used to identify a threshold that improved the balance between precision and recall.

The selected threshold was:

**0.70**

### Hyperparameter Tuning

GridSearchCV was used to tune the Logistic Regression regularization parameter:

`C = [0.01, 0.1, 1, 10, 100]`

The best value identified during cross-validation was:

**C = 100**

---

## 🏆 Final Model

The final system uses an optimized **Logistic Regression classifier**.

Performance on the untouched test set:

- **Accuracy:** 86.16%
- **Precision:** 33.95%
- **Recall:** 37.48%
- **F1 Score:** 35.63%

Confusion matrix:

```text
                 Predicted
                 0       1

Actual 0       8233    745
Actual 1        639    383
```

The model successfully identifies a substantially larger portion of records requiring review than the original baseline model.


## ☁️ Azure Machine Learning Deployment
The trained model was deployed using Microsoft Azure Machine Learning.
Azure components used include:
Azure Machine Learning Workspace
Azure ML Compute Instance
Azure ML Model Registry
Managed Online Endpoint
Custom inference environment
REST-style model inference


Deployment Architecture


Input Data
    ↓
Data Preprocessing
    ↓
Machine Learning Model
    ↓
Probability Prediction
    ↓
0.70 Decision Threshold
    ↓
Review Required / No Review Required

The model is exposed through an Azure ML managed online endpoint, allowing new records to be submitted for prediction.


🚀 Example Prediction
A sample high-risk record was submitted to the deployed endpoint.
Example response:
```
[
  {
    "needs_review": 1,
    "review_probability": 0.993,
    "decision": "REVIEW REQUIRED"
  }
]
```
This demonstrates that the trained model can be used as an inference service rather than only as a local notebook model.

🛠️ Technologies
Python
Pandas
NumPy
scikit-learn
Jupyter Notebook
Microsoft Azure Machine Learning
Azure ML Model Registry
Azure ML Managed Online Endpoints
GitHub


📁 Repository Contents
Data_Quality_Agent.ipynb
Complete machine learning development notebook containing data preparation, model training, evaluation, optimization, and Azure ML deployment workflow.
model_comparison.csv
Model evaluation results comparing the different classification approaches.


💼 Business Impact
A production version of this system could help organizations:
Prioritize records for human review
Reduce unnecessary manual inspection
Improve data-quality workflows
Identify potential quality issues earlier
Provide analysts with cleaner downstream data
The system is designed as a decision-support tool, with human review remaining part of the quality-control process.



🔮 Future Improvements
Potential next steps include:
Testing additional classification algorithms
Improving minority-class detection
Evaluating additional features
Performing larger-scale stress testing
Monitoring model performance after deployment
Retraining the model as new labeled data becomes available
Adding automated data-quality dashboards
Integrating the endpoint with a production data pipeline


👤 Project Author
EvanK024 -- Evan Middleton
This project demonstrates practical experience with data analysis, supervised machine learning, model evaluation, optimization, and cloud-based ML deployment.
