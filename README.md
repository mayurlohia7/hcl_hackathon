
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

2. Input (X) and Output (y)
Input Features (X)

Only features available at purchase time (avoiding future leakage):

Product information (category, size, type)

Pricing & discount details

Customer details (if available)

Order context (date/time, day of week, season)

Purchase channel (web / mobile / store)

Shipment preferences

Output Target (y)

Binary label:

1 → Order Returned

0 → Order Not Returned

Model also predicts a probability score (0 to 1).

3. End-to-End Project Pipeline
3.1 Data Ingestion

Loaded dataset using pandas

Inspected datatypes and structure
Functions used: pd.read_csv(), df.head(), df.info()

3.2 Data Preprocessing

Performed:

Handling missing values

Removing duplicates

Fixing datatypes

Dropping irrelevant/leakage columns

Standardizing categorical values

Libraries: pandas, scikit-learn
Tools: SimpleImputer, ColumnTransformer

3.3 Exploratory Data Analysis (EDA)

Studied patterns such as:

Class imbalance (Returned vs Not Returned)

Category-wise return rates

Relation of price/discount with returns

Seasonal or day-of-week trends

Libraries: matplotlib, seaborn

3.4 Feature Engineering

Created meaningful features:

Time-based (month, day, season)

Discount percentage

Price bins

Encoded categorical values

Tools: OneHotEncoder, StandardScaler

3.5 Feature Selection

Performed by Poonam

Removed highly correlated columns

Removed identifiers

Eliminated low-importance features
Tools: .corr(), SelectKBest

3.6 Model Selection

Models evaluated:

Logistic Regression

Random Forest

Gradient Boosting

XGBoost (optional)

Metrics used:

Accuracy

Precision

Recall

F1-score

ROC-AUC

 Actual Project Journey (Important for Understanding the Final Model)
Initial Attempt: Regression Approach (Failed)

Originally, we tried predicting return_rate using regression models.
However:

All major regressors gave negative R² scores

Model performed worse than predicting the mean

Regression clearly did not fit the problem

 Decision: Switch to Classification

Pivot to Classification

We converted return_rate into a binary target using a threshold:

Above threshold → Returned (1)

Below threshold → Not Returned (0)

But a new problem appeared:

⚠ Poor Precision & Recall

Despite decent accuracy, both precision and recall were low because:

Dataset had very few actual return cases

Strong class imbalance

Model could not learn return patterns well

Result: Limited performance due to dataset limitations, not model architecture.

3.7 Model Training

Performed by Bhumika & Mayur

Train–test split

Model pipeline (preprocessing + model)

Hyperparameter tuning
Functions used: fit(), predict(), predict_proba()

3.8 Model Packaging

Final chosen classifier saved using pickle

pickle.dump(model, open("model.pkl", "wb"))


This pickle file is later loaded in Streamlit.

3.9 Streamlit UI (Frontend + Backend Combined)

Developed by Mohit

Streamlit handles:

User input forms

Data preprocessing consistency

Calling the ML model for predictions

Displaying the probability and final prediction

Components used:
st.number_input(), st.selectbox(), st.button(), model.predict_proba()

3.10 Final Output

Streamlit displays:

Return Probability (0–1)

Prediction (Returned / Not Returned)

Optional Recommendations



4.SYSTEM ARCHITECTURE DIAGRAM:

        ┌────────────────────┐
        │   Data Source      │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │  Data Preprocessing │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │ Feature Engineering │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │   Train-Test Split  │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │   Model Training    │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │ Model Evaluation    │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │  Model Pickle File  │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │  Streamlit Backend  │
        └─────────┬──────────┘
                  │
        ┌─────────▼──────────┐
        │   Streamlit UI      │
        └─────────────────────┘




5.ROLE SPLIT IN THE TEAM :

Bhumika Mishra : Model training

Poonam Kashyap : Evaluation metric and  Feature selection

Aman : data preprocessing, feature engineering

Mayur : Hyperparameter training

Mohit : Deployment, Frontend


















