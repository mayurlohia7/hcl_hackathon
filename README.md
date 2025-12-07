
HCL HACKATHON
Problem Statement : Predicting Product return  risk in retail using Classification

Team Members Details :
_______________________________________________________
| Team Member Name | Roll no  | Email Id              |
_______________________________________________________
| Bhumika Mishra   | 12212002 | 12212002@nitkkr.ac.in |
_______________________________________________________
| Poonam Kashyap   | 12212003 | 12212003@nitkkr.ac.in |
_______________________________________________________
| Mohit            | 12213058 | 12213058@nitkkr.ac.in |
_______________________________________________________
| Mayur            | 12113072 | 12113072@nitkkr.ac.in |
_______________________________________________________
| Aman             | 12211149 | 12211149@nitkkr.ac.in |
_______________________________________________________



1. Machine Learning Problem Definition
The objective of this project is to predict whether an order (or an item within an order) will be returned by a customer in a retail or e-commerce environment.
This is framed as a supervised binary classification problem.
Retailers often suffer financial losses due to high product return rates. By predicting return risk at the time of purchase, businesses can identify high-risk orders early and take preventive measures (e.g., validating customer details, showing sizing guides, recommending alternatives, or tightening return policies).


Input (X)
These are the features available at order time only, to avoid future leakage.
 Examples include:
 -> Product information (category, size, type)
 -> Pricing & discount details
 -> Customer details (if available)
 -> Order context (date/time, day of week, season)
 -> Purchase channel (web / mobile / store)
 -> Shipment/delivery preferences



Output (y)
A binary label representing return risk:
1 → Order was returned or cancelled!

0 → Order was not returned

This label is derived from the dataset’s status column (e.g., returned, is_returned, status, etc.).

2. Expected Target Value
The project predicts a binary output:
Predicted Value
Meaning
0 : The order is not likely to be returned
1 : The order is likely to be returned


The model outputs both:
A class prediction (0 or 1)

A probability score (0.0–1.0) representing return risk
Example:
 0.72 return probability → High likelihood of return.


3. End-to-End Project Pipeline (For README)
3.1 Data Ingestion
Load a public retail dataset from Kaggle or other open-source portals.
Import the data into the notebook using pandas.
Inspect the first few rows, datatypes, and column descriptions.


Libraries: pandas
 Key functions: pd.read_csv(), df.head(), df.info()

 3.2 Data Preprocessing
Tasks include:
 -> Handling missing values
 -> Removing duplicates
 -> Fixing incorrect datatypes
 -> Dropping irrelevant or leakage columns
 -> Standardizing categorical values


Libraries: pandas, scikit-learn
 Key functions: SimpleImputer, .dropna(), .astype(), ColumnTransformer

3.3 Exploratory Data Analysis (EDA)
Perform analysis to understand patterns:
 -> Class distribution (returned vs not returned)
 -> Product category return rates
 -> Price vs return probability
 -> Channel return behavior
 -> Seasonal or day-of-week trends
 
Libraries: matplotlib, seaborn
 Key functions: sns.countplot(), sns.boxplot(), plt.hist()



3.4 Feature Engineering
Create meaningful new features such as:
 -> Time-based features (day, month, season)
 -> Discount percentage
 -> Price bins
 -> Encoded categorical variables
Libraries: pandas, scikit-learn
 Key functions:
 OneHotEncoder, StandardScaler, pd.to_datetime()



3.5 Feature Selection
Identify relevant columns for modeling and remove:
Highly correlated features
Unnecessary identifiers
Post-purchase information (to avoid leakage)


Libraries: pandas, scikit-learn
 Key functions:
 .corr(), SelectKBest

3.6 Model Selection
Evaluate multiple models:
Logistic Regression
Random Forest
Gradient Boosting
XGBoost (optional)


Choose the model that balances:
Precision
Recall
F1-score
ROC-AUC


Libraries: scikit-learn
 Key functions:
 RandomForestClassifier, LogisticRegression, train_test_split
3.7 Model Training
Split dataset into train/test
Train the model pipeline (preprocessing + model)
Evaluate using classification metrics


Libraries: scikit-learn
 Key functions:
 fit(), predict(), predict_proba(), classification_report

3.8 Model Packaging
Save the final trained model using:
Libraries: pickle
 Key functions:
 dump(model, "model.pkl")
This file will be used inside the Streamlit UI.


3.9 Streamlit UI Application (Frontend + Backend Combined)
Streamlit will:
Accept user inputs (price, channel, category, etc.)
Pass inputs to the loaded model
Display return risk probability
Provide interpretation or explanation

Libraries: streamlit, pandas, joblib
 Key functions:
 st.number_input(), st.selectbox(), st.button(), load(), predict_proba()
Streamlit acts as both:
✔ UI (frontend)
 ✔ Model inference engine (backend)
So no separate API is required.



3.10 Final Output
The app outputs:
Return risk probability (0–1)


Model prediction (Returned / Not Returned)
Optional: insights or recommendations




SYSTEM ARCHITECTURE DIAGRAM:



Steps :
Data Source 
Public retail/order dataset chosen by your team.
Data Pre‑processing
Data cleaning (handle missing values, remove duplicates, fix data types).
Feature engineering (derive features like day of week, season, price ranges).
Encoding categorical variables and scaling numeric features.
     3)Train–Test Split
Split preprocessed data into training and test (and optionally validation) sets.
    4) Model Training (Logistic Regression)
Train logistic regression classifier on training set.


Tune hyperparameters and handle class imbalance if needed.
5) Model Evaluation
Evaluate on test set using Accuracy, Precision, Recall, F1‑score, ROC‑AUC, etc.
6) Model Serialization
Save the trained model and any preprocessors (encoders, scalers) as files (e.g., pickle).
7) Streamlit Backend
Load saved model and preprocessors.
Define a prediction function that takes new order details, applies the same preprocessing, and calls the logistic regression model.
8) Streamlit Frontend UI
Input widgets for product, price, order, and customer features.
“Predict Return Risk” button that shows probability and class (Return / No Return).


ROLE SPLIT IN THE TEAM :

Bhumika Mishra : Model training

Poonam Kashyap : Evaluation metric and  Feature selection

Aman : data preprocessing, feature engineering

Mayur : Hyperparameter training

Mohit : Deployment, Frontend


















