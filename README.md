
📊 Exploratory Data Analysis (EDA) Report
📌 Dataset Overview

Total Records: 500

Total Features: 21

Data Source: Smart Loan Recovery System Dataset

Objective: Predict borrower default risk & support recovery planning

🧩 Feature Types
Feature Type	Columns
🔢 Numerical	Age, Monthly_Income, Num_Dependents, Loan_Amount, Loan_Tenure, Interest_Rate, Collateral_Value, Outstanding_Loan_Amount, Monthly_EMI, Num_Missed_Payments, Days_Past_Due, Collection_Attempts
🔠 Categorical	Gender, Employment_Type, Loan_Type, Payment_History, Recovery_Status, Collection_Method, Legal_Action_Taken, Borrower_ID, Loan_ID
🧪 Data Quality Check

❗ Missing Values: 0 (No nulls found — dataset is clean)

🔁 Duplicates: Checked & handled if present

🏷 Data Types: Correctly assigned (categorical/object & numeric values separated properly)

📈 Numerical Feature Insights

Loan & Income values vary widely across borrowers

Higher Num_Missed_Payments and Days_Past_Due indicate repayment risk

Outstanding Loan Amount, EMI & Loan Amount show financial load pattern

Age and Income distributions reflect typical working population borrower base

🔡 Categorical Feature Insights

Gender, Employment Type, Loan Type, Payment History show diverse borrower categories

Payment_History likely influences default behavior

Recovery_Status acts as our Target Variable

🔥 Correlation Highlights

📌 Strong indicators of default risk:

Num_Missed_Payments

Days_Past_Due

Payment_History

Outstanding_Loan_Amount

Loan_Amount ↔ Monthly_EMI show strong positive relationship

Income-Loan balance influences repayment capability

📝 Key Observations

Dataset is clean, structured, and modeling-ready

No missing data → smooth preprocessing step

Recovery_Status selected as Target for classification

Ideal for Logistic Regression binary prediction model
