# 🏦 Loan Approval Prediction using Machine Learning

## 📌 Overview

The **Loan Approval Prediction** project is a Machine Learning application that predicts whether a loan application is likely to be **Approved** or **Rejected** based on applicant information. The project involves data preprocessing, handling class imbalance, training multiple machine learning models, and selecting the best-performing model for prediction.

The entire project was developed and evaluated in **Google Colab** using Python and popular machine learning libraries.

---

## 🚀 Features

- 📊 Data preprocessing and cleaning
- ⚖️ Handles imbalanced data using **SMOTE**
- 📈 Feature scaling using **StandardScaler**
- 🤖 Trains and compares multiple Machine Learning models
- 🏆 Selects the best-performing model
- 📉 Confusion Matrix visualization
- 📊 ROC Curve and AUC Score evaluation
- 📋 Classification Report
- 📌 Feature Importance visualization using Random Forest

---

## 🧠 Machine Learning Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier ⭐ (Best Model)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## 📂 Dataset

The project uses a **Loan Approval Dataset** containing applicant details such as:

- Applicant Income
- Loan Amount
- Credit Score
- Employment Status
- Education
- Loan Term
- Existing Debts
- Other financial attributes

**Target Variable**

- Loan Approved
  - 1 → Approved
  - 0 → Rejected

---

## 🛠️ Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn

---

## 📊 Project Workflow

1. Load the dataset
2. Data preprocessing
3. Split training and testing data
4. Handle class imbalance using SMOTE
5. Standardize numerical features
6. Train multiple ML models
7. Compare model performance
8. Select the best model (Random Forest)
9. Evaluate using:
   - Accuracy
   - Classification Report
   - Confusion Matrix
   - ROC-AUC Score
10. Visualize Feature Importance

---

## 📈 Model Evaluation

Evaluation metrics used:

- Accuracy Score
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

## 📷 Project Output

### Confusion Matrix

> Add your confusion matrix screenshot here.

```
images/confusion_matrix.png
```

### ROC Curve

> Add your ROC curve screenshot here.

```
images/roc_curve.png
```

### Feature Importance

> Add your feature importance chart here.

```
images/feature_importance.png
```

---

## 📁 Project Structure

```
Loan-Approval-Prediction/
│
├── Loan_approval_prediction.ipynb
├── loan_approval.csv
├── README.md
├── .gitignore
└── images/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

---

## ▶️ How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/Loan-Approval-Prediction.git
```

2. Navigate to the project

```bash
cd Loan-Approval-Prediction
```

3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

4. Open the Jupyter Notebook or Google Colab.

5. Run all cells sequentially.

---

## 💡 Future Improvements

- Deploy as a Flask or Gradio web application
- Hyperparameter tuning
- Cross-validation
- Explain predictions using SHAP or LIME
- Real-time loan prediction interface
- Model deployment using Streamlit

---

## 👨‍💻 Author

**Dharani Rajesh**

Aspiring AI & Machine Learning Developer passionate about building practical ML applications and solving real-world problems.

---

## ⭐ If you found this project useful, don't forget to Star this repository!
