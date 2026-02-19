# Loan Default Prediction

A machine learning pipeline that predicts whether a loan applicant will default, using an ensemble of Random Forest and XGBoost classifiers optimized for recall.

---

## Problem Statement

Loan default prediction is a class imbalance problem — defaulters are a small minority of applicants. A standard accuracy-focused model will simply learn to predict "no default" for everyone. This project addresses that by using SMOTE oversampling, class weighting, and threshold tuning to maximize the detection of real defaulters.

---

## Dataset

**File:** `Loan_default.csv`

| Feature | Description |
|---|---|
| `Age` | Applicant age |
| `Income` | Annual income |
| `LoanAmount` | Requested loan amount |
| `CreditScore` | Credit score |
| `MonthsEmployed` | Months at current job |
| `NumCreditLines` | Number of open credit lines |
| `InterestRate` | Loan interest rate |
| `LoanTerm` | Loan duration in months |
| `DTIRatio` | Debt-to-income ratio |
| `Education` | Education level |
| `EmploymentType` | Employment status |
| `MaritalStatus` | Marital status |
| `HasMortgage` | Whether applicant has a mortgage |
| `HasDependents` | Whether applicant has dependents |
| `LoanPurpose` | Purpose of loan |
| `HasCoSigner` | Whether a co-signer is present |
| `Default` | Target — 1 if defaulted, 0 if not |

**Class distribution:** ~88% non-default, ~12% default (heavily imbalanced)

---

## Project Structure

```
├── Loan_default.csv
├── loan_default_prediction.ipynb
├── loan_default_model.pkl
├── best_threshold.pkl
└── README.md
```

---

## Pipeline Overview

### 1. Preprocessing
- Dropped `LoanID` (identifier, no predictive value)
- Encoded binary columns (`HasMortgage`, `HasDependents`, `HasCoSigner`) as 0/1
- One-hot encoded categorical columns (`Education`, `EmploymentType`, `MaritalStatus`, `LoanPurpose`)

### 2. Feature Engineering
Three new features were created to improve signal:

| Feature | Formula | Purpose |
|---|---|---|
| `LoanToIncome` | `LoanAmount / (Income + 1)` | Debt burden ratio |
| `CreditLinesPerYear` | `NumCreditLines / (MonthsEmployed/12 + 1)` | Credit usage rate |
| `RiskInteraction` | `InterestRate × DTIRatio` | Combined financial stress |

Original columns (`LoanAmount`, `NumCreditLines`, `MonthsEmployed`) were dropped after engineering to remove redundancy.

### 3. Handling Class Imbalance
- **SMOTE** (Synthetic Minority Oversampling Technique) was applied to the training set to balance the classes from ~8:1 to 1:1
- `scale_pos_weight` was passed to XGBoost to further penalize missing defaulters

### 4. Models
A soft-voting ensemble of two tuned models:

- **Random Forest** — tuned with `RandomizedSearchCV`, `class_weight='balanced'`, optimized for recall
- **XGBoost** — tuned with `RandomizedSearchCV`, `scale_pos_weight` set to class ratio, optimized for recall

Logistic Regression was excluded from the ensemble as it underperformed on this non-linear problem.

### 5. Threshold Tuning
Instead of using the default 0.5 classification threshold, the pipeline tests thresholds from 0.1 to 0.9 and selects the threshold that maximizes recall while maintaining a minimum precision of 0.18, giving the best tradeoff for catching real defaulters.

---

## Results

| Metric | Class 0 (No Default) | Class 1 (Default) |
|---|---|---|
| Precision | 0.95 | 0.18 |
| Recall | 0.56 | **0.76** |
| F1-Score | 0.70 | 0.29 |

**ROC-AUC: 0.71**

The model catches **76% of actual defaulters**, reducing missed defaults from ~2,900 to ~1,437 compared to a naive baseline. The tradeoff is lower precision on defaults, meaning some non-defaulters are flagged — acceptable in a lending context where missing a real default is more costly.

---

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
xgboost
imbalanced-learn
scipy
joblib
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib scikit-learn xgboost imbalanced-learn scipy joblib
```

---

## Usage

### Run the notebook
```bash
jupyter notebook loan_default_prediction.ipynb
```

### Load the saved model
```python
import joblib

model     = joblib.load('loan_default_model.pkl')
threshold = joblib.load('best_threshold.pkl')

# Predict on new data (must be preprocessed the same way)
probs  = model.predict_proba(new_data)[:, 1]
preds  = (probs >= threshold).astype(int)
```

---

## Key Design Decisions

**Why SMOTE instead of just class weights?**
Class weights adjust the loss function but the model still trains on the original imbalanced distribution. SMOTE creates synthetic minority examples so the model actually sees more diverse defaulter patterns during training.

**Why remove Logistic Regression from the ensemble?**
LR is a linear model and cannot capture the non-linear relationships between features and default risk. Including it was dragging down the ensemble's performance.

**Why optimize for recall over F1 or accuracy?**
In a lending context, a missed default (false negative) is more costly than a false alarm (false positive). Optimizing for recall ensures the model prioritizes catching real defaulters even at the cost of some precision.

---

## Author

Built as a binary classification project for loan default risk modeling.
