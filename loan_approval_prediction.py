import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                            roc_auc_score, roc_curve, make_scorer, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('loan_approval_dataset.csv')

# ======================
# EDA and Preprocessing
# ======================

# Correcting mismatched data types
try:
    data['loan_int_rate'] = pd.to_numeric(data['loan_int_rate'], errors='coerce')
except KeyError:
    st.warning("'loan_int_rate' column not found.")

# Outlier Handling (using IQR)
def remove_outliers_iqr(df, column, lower_quantile=0.05, upper_quantile=0.95):
    Q1 = df[column].quantile(lower_quantile)
    Q3 = df[column].quantile(upper_quantile)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

numerical_cols_for_outlier = ['person_age', 'person_income', 'person_emp_exp',
                              'loan_amnt', 'loan_int_rate', 'loan_percent_income',
                              'cb_person_cred_hist_length', 'credit_score']

for col in numerical_cols_for_outlier:
    data = remove_outliers_iqr(data, col)

# Remove sensitive attribute
try:
    data.drop('person_gender', axis=1, inplace=True)
except KeyError:
    st.warning("'person_gender' column not found.")

# Feature Engineering
def feature_engineer(df):
    if not pd.api.types.is_string_dtype(df['person_education']):
        df['person_education'] = df['person_education'].astype(str)

    education_map = {
        'High School': 1,
        'Associate': 2,
        'Bachelor': 3,
        'Master': 4
    }
    df['education_to_income'] = df['person_income'] / df['person_education'].map(education_map).fillna(0)
    df['rent_vs_income'] = np.where(df['person_home_ownership'] == 'RENT',
                                     df['loan_amnt'] / df['person_income'],
                                     0)
    df['employment_stability'] = df['person_emp_exp'] / (df['person_age']+0.001)
    intent_target_mean = df.groupby('loan_intent')['loan_status'].mean()
    df['loan_intent_weighted'] = df['loan_intent'].map(intent_target_mean)
    return df

data = feature_engineer(data)

# Select final features (after engineering)
features = [
    'person_age',
    'person_income',
    'education_to_income',
    'rent_vs_income',
    'employment_stability',
    'loan_intent_weighted',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score',
    'previous_loan_defaults_on_file'
]

# Handle potentially missing features after transformations
for feature in features:
    if feature not in data.columns:
        st.warning(f"Feature '{feature}' not found in data after feature engineering.")
        features.remove(feature)

target = 'loan_status'

# Convert previous_loan_defaults to binary
data['previous_loan_defaults_on_file'] = data['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})

# Impute missing values before splitting to prevent data leakage
for col in data.columns:
    if data[col].isnull().any():
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].mean())

# Split data
X = data[features]
y = data[target]

# Handle potential infinity values
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    if X[col].isnull().any():
        if X[col].dtype == 'object':
            X[col] = X[col].fillna(X[col].mode()[0])
        else:
            X[col] = X[col].fillna(X[col].mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE after splitting
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Store column names before the ColumnTransformer
numerical_features_before_ohe = X.select_dtypes(include=np.number).columns.tolist()
categorical_features_before_ohe = X.select_dtypes(exclude=np.number).columns.tolist()

# Create separate numerical and categorical dataframes
X_train_numerical = X_train_resampled[numerical_features_before_ohe].copy()
X_test_numerical = X_test[numerical_features_before_ohe].copy()
X_train_categorical = X_train_resampled[categorical_features_before_ohe].copy()
X_test_categorical  = X_test[categorical_features_before_ohe].copy()

# Define numerical and categorical features dynamically
numerical_features = X_train_numerical.columns.tolist()
categorical_features = X_train_categorical.columns.tolist()

# Numerical transformer
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical transformer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Transform data using the preprocessor
X_train_transformed = preprocessor.fit_transform(X_train_numerical.join(X_train_categorical))
X_test_transformed = preprocessor.transform(X_test_numerical.join(X_test_categorical))

# Model Training & Tuning
def train_and_evaluate(model, param_grid, X_train, y_train, X_test, y_test, name):
    st.write(f"\nTraining and Tuning {name}:")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(balanced_accuracy_score), cv=kf, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    st.write(f"Best parameters for {name}:", grid_search.best_params_)
    st.write(f"Best balanced accuracy score for {name}:", grid_search.best_score_)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    st.write(f"\nTest Set Performance for {name}:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, y_proba):.2f}")
    st.write(classification_report(y_test, y_pred))
    return best_model, y_proba

# Random Forest
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 10],
    'min_samples_leaf': [5, 10]
}
rf_best_model, rf_proba = train_and_evaluate(rf_model, rf_param_grid, X_train_transformed, y_train_resampled, X_test_transformed, y_test, "Random Forest")

# XGBoost
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.01]
}
xgb_best_model, xgb_proba = train_and_evaluate(xgb_model, xgb_param_grid, X_train_transformed, y_train_resampled, X_test_transformed, y_test, "XGBoost")

# Model Comparison
st.write("### ROC Curve Comparison")
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, rf_proba)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_proba):.2f})')
fpr, tpr, _ = roc_curve(y_test, xgb_proba)
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc_score(y_test, xgb_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
st.pyplot(plt)

# User Input & Prediction
def predict_loan_approval(model, preprocessor, numerical_features, categorical_features):
    st.write("### Enter user data for loan approval prediction:")
    user_data = {}
    for feature in numerical_features:
        user_data[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)
    for feature in categorical_features:
        user_data[feature] = st.text_input(f"Enter value for {feature}:", value="")
    
    if st.button("Predict"):
        user_df = pd.DataFrame([user_data])
        user_transformed = preprocessor.transform(user_df[numerical_features + categorical_features])
        prediction = model.predict(user_transformed)
        probability = model.predict_proba(user_transformed)[:, 1][0]
        st.write(f"\nLoan {'Approved' if prediction[0] == 1 else 'Denied'}")
        st.write(f"Approval Probability: {probability:.2%}")

# Call User Input and Prediction
predict_loan_approval(rf_best_model, preprocessor, numerical_features, categorical_features)