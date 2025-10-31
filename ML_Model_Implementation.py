import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- Model Training Setup  ---

def train_model():
    file_names = {
        'df_meta': 'Metadata.csv',
        'df_demo': 'Demographic _102.csv',
        'df_psycho': 'Psychometrics.csv',
        'df_task': 'TaskOrder.csv',
    }

    def load_and_clean(file_path):
        df = pd.read_csv(file_path, dtype={'ID': str}, na_values=['NA', 'N/A', ''])
        df['ID'] = df['ID'].astype(str).str.split('.').str[0].str.strip()
        return df

    dataframes = {name: load_and_clean(path) for name, path in file_names.items()}
    df_merged = dataframes['df_task'].copy()
    dfs_to_merge = [dataframes['df_meta'], dataframes['df_demo'], dataframes['df_psycho']]

    for i, df_right in enumerate(dfs_to_merge):
        cols = ['ID']
        if 'df_meta' in file_names and df_right is dataframes['df_meta']:
            cols.append('Bonus')
        elif 'df_demo' in file_names and df_right is dataframes['df_demo']:
            # Exclude Age/Income due to high missing values
            cols.extend(['Gender', 'Education'])
        elif 'df_psycho' in file_names and df_right is dataframes['df_psycho']:
            psycho_cols = [col for col in df_right.columns if col.startswith('dssQuestion')]
            cols.extend(psycho_cols)

        df_merged = pd.merge(
            df_merged,
            df_right[cols].drop_duplicates(subset=['ID']),
            on='ID',
            how='inner',
            suffixes=('', f'_dup{i + 1}')
        )

    # --- Feature and Model Prep ---
    Y = df_merged['Type']
    numerical_features = ['EarnedPoints', 'OptimalPoints', 'Time', 'Correlation', 'Bonus']
    categorical_features = ['Gender', 'Education', 'CorrelationCondition']

    # Calculate feature
    dss_cols = [col for col in df_merged.columns if col.startswith('dssQuestion')]
    df_merged['DSS_Mean'] = df_merged[dss_cols].mean(axis=1)
    numerical_features.append('DSS_Mean')

    X = df_merged[numerical_features + categorical_features]
    X_train, _, Y_train, _ = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # --- Preprocessing Pipeline ---
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # --- Random Forest Pipeline ---
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # the model on the full training set
    rf_pipeline.fit(X_train, Y_train)

    return rf_pipeline, numerical_features, categorical_features


# ---  CLI Prediction Function ---

def predict_task_category(model, num_features, cat_features):
    print("\n" + "=" * 50)
    print("TASK CATEGORIZATION PREDICTION INTERFACE (Random Forest)")
    print("=" * 50)

    input_data = {}

    # Get Numerical Inputs
    print("\nPlease enter values for the **Task/Participant Numeric Features**:")
    for feature in num_features:
        while True:
            try:
                # DSS_Mean is calculated; skip raw DSS questions
                if feature == 'DSS_Mean':
                    print("  NOTE: DSS_Mean is the average psychometric score.")
                    break
                value = float(input(f"  Enter {feature}: "))
                input_data[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a number.")

    # Get Categorical Inputs
    print("\nPlease enter values for the **Participant Categorical Features**:")
    for feature in cat_features:
        value = input(f"  Enter {feature}: ")
        input_data[feature] = value

    # Create a DataFrame from the user input
    # It must be a DataFrame for the preprocessor to work correctly
    new_data = pd.DataFrame([input_data])

    # Calculate DSS_Mean for the single input row
    if 'DSS_Mean' not in new_data.columns:
        new_data['DSS_Mean'] = np.nan

    all_features = num_features + cat_features
    new_data = new_data[all_features]

    # prediction
    prediction = model.predict(new_data)[0]

    print("\n" + "-" * 50)
    print("🎯 PREDICTION RESULT:")
    print(f"Based on the input features, the predicted Task Category is: **{prediction.upper()}**")
    print("-" * 50)


# --- Main Execution ---
if __name__ == "__main__":
    # Train the model (takes a moment)
    print("Training Random Forest Model... (This runs once)")
    trained_rf_pipeline, num_feats, cat_feats = train_model()
    print("Training complete. Model ready for predictions.")

    # Run the CLI interface
    predict_task_category(trained_rf_pipeline, num_feats, cat_feats)