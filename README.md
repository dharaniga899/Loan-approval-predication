# 🏦 Loan Approval Prediction using Machine Learning

A Machine Learning application that predicts whether a loan application is likely to be **Approved** or **Rejected** based on applicant information. The project demonstrates the complete machine learning workflow, including data preprocessing, feature analysis, handling class imbalance using **SMOTE**, training multiple classification models, model evaluation, and selecting the best-performing model.

The project was developed and evaluated entirely in **Google Colab** using Python and the Scikit-learn ecosystem, and deployed as a live interactive web app using **Gradio** on **Hugging Face Spaces**.

---

## 🌐 Live Demo

👉 **[Try the Loan Approval Predictor](https://huggingface.co/spaces/dharaniga123/loan-approval-prediction)**

Enter applicant details (income, credit score, loan amount, years employed) using the sliders and get an instant Approved / Not Approved prediction along with approval probability and smart recommendations.

---

## 📌 Overview

Financial institutions receive thousands of loan applications every day. Evaluating each application manually is time-consuming and may lead to inconsistent decisions.

This project uses Machine Learning techniques to automate the loan approval process by analyzing applicant information and predicting whether a loan should be approved. Multiple classification algorithms were trained and compared, with **Random Forest** achieving the best overall performance.

---

## 🚀 Features

- 📊 Data preprocessing and cleaning
- ⚖️ Class imbalance handling using SMOTE
- 📈 Feature scaling with StandardScaler
- 🤖 Training multiple Machine Learning models
- 🏆 Automatic best model selection
- 📉 Confusion Matrix visualization
- 📈 ROC Curve and AUC evaluation
- 📋 Classification Report
- 🌟 Feature Importance visualization
- 📊 Correlation Heatmap
- 📈 Exploratory Data Analysis (EDA)
- 🎛️ Interactive Gradio web interface with live sliders
- 💡 Smart, rule-based recommendations for applicants

---

## 🧠 Machine Learning Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier ⭐ *(Best Performing Model)*
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## 📂 Dataset

The project uses a Loan Approval dataset containing applicant information such as:

- Applicant Income
- Credit Score
- Loan Amount
- Years Employed
- Loan Approval Status

### 🎯 Target Variable

| Value | Meaning |
|--------|----------|
| 1 | Approved |
| 0 | Not Approved |

---

## 🛠️ Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Gradio (web interface)
- Hugging Face Spaces (deployment)

---

## 📊 Project Workflow

1. Load the dataset
2. Data preprocessing and cleaning
3. Exploratory Data Analysis (EDA)
4. Handle class imbalance using SMOTE
5. Standardize numerical features
6. Split data into training and testing sets
7. Train multiple Machine Learning models
8. Compare model performance
9. Select the best-performing model (Random Forest)
10. Evaluate using:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC Score
    - Confusion Matrix
11. Analyze Feature Importance
12. Build an interactive Gradio interface for live predictions
13. Deploy on Hugging Face Spaces

---

## 📈 Model Evaluation

The Random Forest model achieved excellent predictive performance using the following evaluation metrics:

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-Score
- ✅ ROC-AUC Score (0.9458)
- ✅ Confusion Matrix

---

## 📷 Project Output

### 📊 Loan Approval Distribution
<img width="755" height="546" alt="loan_approval_distribution" src="https://github.com/user-attachments/assets/a5298bcd-0609-4e62-b6e2-102d3cb8a967" />

---

### 📈 Feature Correlation Heatmap
<img width="992" height="595" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/39b7d760-844e-4808-b1ae-93c0a3e69028" />

---

### ✅ Confusion Matrix
<img width="800" height="623" alt="confusion_matrix" src="https://github.com/user-attachments/assets/010737e1-031e-4f1d-acbe-0efdf508e6dc" />

---

### 🌟 Feature Importance (Random Forest)
<img width="948" height="646" alt="feature_importance" src="https://github.com/user-attachments/assets/6f20524d-5c61-4a3a-a8e9-4ce273564c93" />

---

### 📉 ROC Curve (AUC = 0.9458)
<img width="815" height="500" alt="roc_curve" src="https://github.com/user-attachments/assets/94f74745-48e2-4242-8f41-c7a9fd47e8e5" />

---

### 🖥️ Live Application Interface
The deployed Gradio app lets users adjust applicant details with sliders and instantly view the approval decision, probability score, and personalized recommendations.

---

## 📁 Project Structure

```
Loan-Approval-Prediction
│
├── Loan approval predication.ipynb          # Backend (data prep, training, evaluation)
├── Front_end_loan_approval_Untitled8.ipynb  # Frontend (Gradio UI)
├── app.py                                   # Deployment file (backend + frontend combined)
├── requirements.txt
├── loan approval datasets.csv
└── README.md
```

---

## ▶️ Getting Started

### Clone the repository
```bash
git clone https://github.com/your-username/Loan-Approval-Prediction.git
```

### Navigate to the project
```bash
cd Loan-Approval-Prediction
```

### Install the required libraries
```bash
pip install -r requirements.txt
```

### Run the notebook (training & analysis)
Open the Jupyter Notebook or Google Colab and run all cells sequentially.

### Run the Gradio app locally
```bash
python app.py
```
Then open the local URL shown in the terminal (usually `http://127.0.0.1:7860`).

---

## 🚀 Deployment

This project is deployed as a live web app using **Gradio** on **Hugging Face Spaces**.

- **Live Space:** https://huggingface.co/spaces/dharaniga123/loan-approval-prediction
- **SDK:** Gradio
- **Files required for deployment:** `app.py`, `requirements.txt`, `loan approval datasets.csv`

---

## 💡 Future Enhancements

- ⚙️ Hyperparameter Optimization
- 📊 Cross Validation
- 🤖 Explain predictions using SHAP or LIME
- 🗄️ Connect to a real database for storing applications
- 📱 Improve mobile responsiveness of the web interface
- 🔐 Add user authentication for lenders/admins

---
## 📸 APPLICATION PREVIEW
 <img width="1917" height="822" alt="Application preview" src="https://github.com/user-attachments/assets/357653c8-f650-403c-8a28-708210fed473" />


 ## 👨‍💻 Author

### **Dharaniga V.R**
Aspiring AI & Machine Learning Developer passionate about building practical Machine Learning solutions that solve real-world problems.


