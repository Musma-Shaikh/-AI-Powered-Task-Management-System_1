import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np

# --- 1. DATA LOADING AND PREPARATION (You must run these steps) ---
# Load the dataset
df = pd.read_csv("TaskOrder.csv")

# Convert relevant columns to numeric and calculate the target variable
for col in ['EarnedPoints', 'OptimalPoints', 'Time', 'Correlation', 'CorrelationCondition']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Define the target variable (y): Performance Ratio
df['PerformanceRatio'] = df['EarnedPoints'] / df['OptimalPoints']

# Drop rows where the target is NA
df.dropna(subset=['PerformanceRatio'], inplace=True)

# Define feature lists based on the dataset structure
numeric_features = [
    'Task1Length', 'Task2Length', 'Task3Length', 'Task4Length', 'Task5Length', 'Task6Length',
    'Task1Deadline', 'Task2Deadline', 'Task3Deadline', 'Task4Deadline', 'Task5Deadline', 'Task6Deadline',
    'Task1Points', 'Task2Points', 'Task3Points', 'Task4Points', 'Task5Points', 'Task6Points',
    'Task1Rank', 'Task2Rank', 'Task3Rank', 'Task4Rank', 'Task5Rank', 'Task6Rank',
    'Correlation'
]
categorical_features = ['VariableLength']

# Select features (X) and target (y)
X = df[numeric_features + categorical_features]
y = df['PerformanceRatio']

# Drop remaining NA values in features for simplicity in this demo
X = X.dropna()
y = y[X.index]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------


# --- 2. PREPROCESSING PIPELINE ---
# Create transformers for numerical and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Impute missing numeric values with the median
    ('scaler', StandardScaler())                    # Scale/Normalize features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # Impute missing categorical values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))                     # Convert to one-hot encoding
])

# Use ColumnTransformer to apply the right transformation to the right columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop any columns not specified
)

# --- 3. MODEL PIPELINE ---
# Create the final pipeline (Preprocessing + Estimator)
# We use RandomForestRegressor as an example model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# --- 4. GRID SEARCH SETUP ---
# Define the hyperparameter grid for the RandomForestRegressor
# The prefix 'regressor__' targets the 'regressor' step in the pipeline
param_grid = {
    'regressor__n_estimators': [50, 100, 200], # Number of trees in the forest
    'regressor__max_depth': [10, 20, None],  # Maximum depth of the tree (None means full depth)
    'regressor__min_samples_split': [2, 5],  # Min samples required to split a node
    'regressor__min_samples_leaf': [1, 2]    # Min samples required at a leaf node
}

# --- 5. GRID SEARCH EXECUTION ---
# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring='r2', # Use R-squared for evaluation
    cv=5,         # Use 5-fold cross-validation
    verbose=2,    # Show progress
    n_jobs=-1     # Use all available CPU cores
)

# Fit the grid search to the training data
# NOTE: This step is computationally intensive and may take a few minutes
# to run depending on your system and dataset size.
# print("Starting GridSearchCV fit...")
# grid_search.fit(X_train, y_train)
# print("GridSearchCV fit complete.")

# --- 6. RESULTS (Access these after fitting) ---
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
# print(f"Best parameters found: {best_params}")
# print(f"Best cross-validation score (R-squared): {best_score:.4f}")