import pandas as pd
import numpy as np
import joblib # Used for model serialization
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # For classification reframe
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime

DATA_FILE = 'Feature_Engineered_Data.csv'
MODEL_FILENAME = 'model_pipeline.joblib'
RANDOM_SEED = 42

# --- Data Loading and Feature Selection ---
print("--- 1. Loading Data and Defining Features ---")

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"ERROR: Data file '{DATA_FILE}' not found. Please ensure it is available.")
    exit()

df.dropna(subset=['Duration_Seconds'], inplace=True)

TARGET_COL = 'Duration_Seconds'
y = df[TARGET_COL]

NUMERICAL_FEATURES = [
    'Bonus',
    'dssQuestion1', 'dssQuestion2', 'dssQuestion3', 'dssQuestion4', 'dssQuestion5',
    'dssQuestion6', 'dssQuestion7', 'dssQuestion8', 'dssQuestion9', 'dssQuestion10',
    'dssQuestion11', 'msShortQuestion1', 'msShortQuestion2', 'msShortQuestion3',
    'msShortQuestion4', 'msShortQuestion5', 'msShortQuestion6',
    'EarnedPoints', 'OptimalPoints', # 'Time' column removed to fix KeyError
    'Feedback_Sentiment_Score', 'Feedback_Subjectivity' # NLP features
]

CATEGORICAL_FEATURES = [
    'Gender',
    'Education',
    'Occupation',
    'DayOfWeek',
    'HourOfDay_Start',
    'Month',
    'Strategy_TFIDF_Dummy' # Binary/Categorical TFIDF feature
]

for col in CATEGORICAL_FEATURES:
    df[col] = df[col].fillna('Missing').astype(str)

X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

# --- Data Splitting ---
print("\n--- 2. Splitting Data into Training and Testing Sets ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
print(f"Training set size: {X_train.shape[0]} samples, Testing set size: {X_test.shape[0]} samples")


# ---  Preprocessing Pipeline (ColumnTransformer) ---
print("\n--- 3. Creating Preprocessing and Model Pipeline ---")
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, NUMERICAL_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ],

    remainder='drop'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)) # Using Random Forest Regressor
])

# Train the model
print(f"Starting model training (RandomForestRegressor) on {X_train.shape[0]} samples...")
start_time = datetime.now()
model_pipeline.fit(X_train, y_train)
end_time = datetime.now()
print(f"Training complete in {(end_time - start_time).total_seconds():.2f} seconds.")


# --- Prediction and Regression Evaluation ---
print("\n--- 5. Evaluating Model Performance (Regression Metrics) ---")

y_pred = model_pipeline.predict(X_test)

def format_duration(seconds):
    """Converts seconds to HH:MM:SS format."""
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    # Only show hours if they are present
    if hours > 0:
        return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
    else:
        return f"{minutes:02d}m {secs:02d}s"

# Calculate Regression Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("-" * 35)
print("Key Performance Indicators (KPIs) - Duration Prediction")
print("-" * 35)
print(f"1. Mean Absolute Error (MAE): {mae:.2f} seconds")
print(f"   (Avg difference between predicted and actual time.)")
print(f"   In Human Time: Avg. error of {format_duration(mae)}")
print(f"\n2. Root Mean Squared Error (RMSE): {rmse:.2f} seconds")
print(f"   (Square root of the average squared difference. Penalizes large errors.)")
print(f"\n3. R-squared (R²): {r2:.4f}")
print(f"   (Proportion of the variance in the target that is predictable from the features. 1.0 is perfect.)")
print("-" * 35)


# --- 6. Model Serialization ---
print(f"\n--- 6. Saving Trained Pipeline to '{MODEL_FILENAME}' ---")
joblib.dump(model_pipeline, MODEL_FILENAME)
print("SUCCESS: Model saved and ready for deployment.")

print("\n" + "=" * 80)
print("--- OPTIONAL REFRAME: Classification Evaluation ---")
print("These metrics are ONLY relevant if you redefine the problem, e.g., to predict success/failure.")
print("=" * 80)

THRESHOLD_TIME = 1200  # 20 minutes (as used in prior HTML snippet)

# Create binary labels: 1 if efficient, 0 otherwise
y_test_binary = (y_test < THRESHOLD_TIME).astype(int)

# Convert regression predictions to binary predictions
y_pred_binary = (y_pred < THRESHOLD_TIME).astype(int)

print(f"Reframed Target: 'Efficient Completion' (Time < {THRESHOLD_TIME} seconds) - Binary Classification")

# Accuracy
accuracy = accuracy_score(y_test_binary, y_pred_binary)
# Precision
precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
# Recall
recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
# F1 Score (Harmonic mean of Precision and Recall)
f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)

print(f"1. Accuracy: {accuracy:.4f}")
print(f"   (Overall percentage of correct predictions.)")
print(f"2. Precision: {precision:.4f}")
print(f"3. Recall: {recall:.4f}")
print(f"   (Of all actual efficient tasks, how many did we correctly predict.)")
print(f"4. F1 Score: {f1:.4f}")
print(f"   (Balanced measure of Precision and Recall.)")
