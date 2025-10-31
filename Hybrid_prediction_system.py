import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random
import sys


# --- 1. DATA PROCESSING AND FEATURE ENGINEERING ---

class DataProcessor:
    """
    Simulates loading, merging, and engineering features from all input CSVs.
    In a real application, you would replace the synthetic methods with actual file loading.
    """

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.analyzer = SentimentIntensityAnalyzer()
        self.df = self._generate_synthetic_data()  # Use synthetic data for simulation

    def _generate_synthetic_data(self):
        """Generates a synthetic dataset simulating the merged data structure."""
        np.random.seed(42)
        N = 500  # Number of simulated users/tasks

        data = {
            'ID': [f'user_{i}' for i in range(N)],

            # Demographic Features (from Demographic_102.csv)
            'Age': np.random.randint(20, 60, N),
            'Gender': np.random.choice(['Male', 'Female', 'Other'], N, p=[0.45, 0.50, 0.05]),
            'Income': np.random.randint(30000, 150000, N),

            # Psychometric Features (from Psychometrics.csv)
            # Simulating two scale scores: Decision Score (dss) and Mindfulness Score (ms)
            'DecisionScore': np.random.uniform(1, 7, N),  # Averaged dssQuestion
            'MindfulnessScore': np.random.uniform(1, 7, N),  # Averaged msShortQuestion

            # Strategy Features (from Strategy.csv and Feedback_102.csv)
            'StrategyText': [random.choice(
                ['Focus on high points', 'Focus on deadlines', 'Random guessing', 'Great game',
                 'Frustrating design, slow speed']) for _ in range(N)],

            # Task Features (from TaskOrder.csv)
            'TaskSetComplexity': np.random.randint(1, 5, N),  # Complexity level of the task set
            'OptimalPoints': np.random.randint(500, 2000, N),
            'EarnedPoints': np.random.randint(300, 1800, N),

            # TARGET VARIABLE (from TaskOrder.csv / Metadata.csv)
            'TaskCompletionTime': np.random.randint(600, 3000, N),  # Time in seconds
        }
        df = pd.DataFrame(data)

        # Create a realistic correlation: high DecisionScore and high OptimalPoints lead to lower (better) time
        df['TaskCompletionTime'] = df['TaskCompletionTime'] - (df['DecisionScore'] * 50) - (df['OptimalPoints'] / 10)
        df['TaskCompletionTime'] = np.maximum(500, df['TaskCompletionTime'])  # Min time of 500s

        return df

    def extract_nlp_features(self, text):
        """Uses VADER to extract sentiment from text features (Strategy/Feedback)."""
        if pd.isna(text):
            return 0.0, 0.0
        vs = self.analyzer.polarity_scores(str(text))
        # Return compound sentiment (overall score) and a simple binary indicator for positive
        return vs['compound'], 1 if vs['compound'] > 0.05 else 0

    def preprocess_input(self, data_to_process):
        """Applies feature engineering to input data, whether for training or prediction."""
        df = data_to_process.copy()

        # 1. NLP Feature Engineering
        df['SentimentScore'], df['PositiveSentiment'] = zip(*df['StrategyText'].apply(self.extract_nlp_features))
        df.drop('StrategyText', axis=1, inplace=True)  # Drop original text column

        # 2. Performance Feature
        # A measure of performance (Ratio of Earned Points to Optimal Points)
        df['PerformanceRatio'] = df['EarnedPoints'] / df['OptimalPoints']

        return df

    def engineer_features(self):
        """Prepares the training data (X and y) and sets up the preprocessor."""
        # Process the full synthetic training data
        processed_df = self.preprocess_input(self.df)

        # Define features (X) and target (y)
        # We need to explicitly exclude 'ID', 'TaskCompletionTime', and the raw points/optimal points
        # since PerformanceRatio is derived from them.
        features = [
            'Age', 'Gender', 'Income', 'DecisionScore', 'MindfulnessScore',
            'TaskSetComplexity', 'SentimentScore', 'PositiveSentiment', 'PerformanceRatio'
        ]

        X = processed_df[features]
        y = self.df['TaskCompletionTime']  # Target remains from the original data

        # Identify categorical and numerical features for preprocessing pipeline
        self.numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_features = ['Gender']

        # Create Preprocessing Pipeline (only scales/encodes, does not run feature creation)
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ],
            remainder='passthrough'
        )

        return X, y


# --- 2. MACHINE LEARNING PREDICTOR ---

class MLPredictor:
    """Trains a Random Forest Regressor to predict task completion time."""

    def __init__(self, X, y, preprocessor, data_processor):
        self.X = X
        self.y = y
        self.preprocessor = preprocessor
        self.data_processor = data_processor  # Keep a reference to the processor
        self.model = None
        self.pipeline = None

    def train(self):
        """Splits data and trains the prediction pipeline."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Build a full pipeline: Preprocessing -> Model
        self.pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])

        self.pipeline.fit(X_train, y_train)
        self.model = self.pipeline  # The pipeline itself acts as the trained model

        # Evaluate performance (R-squared score on test set)
        score = self.model.score(X_test, y_test)
        print(f"\n[ML Predictor] Random Forest R-squared Score (Test Data): {score:.4f}")

    def predict(self, new_data_unprocessed):
        """
        Processes new data through feature engineering FIRST, then makes a prediction.
        :param new_data_unprocessed: Data (dict or DataFrame) without engineered features.
        :return: Predicted time (float).
        """
        # 1. Ensure input is DataFrame
        if isinstance(new_data_unprocessed, dict):
            new_data_unprocessed = pd.DataFrame([new_data_unprocessed])

        # 2. FEATURE ENGINEERING (The step that was missing before the predict call)
        new_data_processed = self.data_processor.preprocess_input(new_data_unprocessed)

        # 3. SELECT ONLY THE FEATURES USED FOR TRAINING
        features_used = [col for col in self.X.columns if col != 'ID']
        X_new = new_data_processed[features_used]

        # 4. PREDICT (The pipeline handles scaling/encoding)
        prediction = self.model.predict(X_new)
        return prediction[0]


# --- 3. RULE-BASED ADJUSTER (HYBRID LOGIC) ---

class RuleBasedAdjuster:
    """Applies human-defined business logic to adjust the ML prediction."""

    def __init__(self, confidence_threshold=0.5):
        # Confidence threshold for the rule system to trigger an override
        self.confidence_threshold = confidence_threshold

    def apply_rules(self, ml_prediction_time, user_data):
        """
        Applies a set of rules to adjust the predicted time.
        :param ml_prediction_time: The raw time prediction from the ML model (in seconds).
        :param user_data: The unprocessed input features for the user (DataFrame or dict).
        :return: Final adjusted time (seconds) and a descriptive reason.
        """
        final_time = ml_prediction_time
        adjustment_reason = "No rules triggered. ML prediction accepted."

        # Convert to single-row DataFrame if input is dict
        if isinstance(user_data, dict):
            user_data = pd.DataFrame([user_data])

        # Extract features for rule checks
        age = user_data['Age'].iloc[0]
        decision_score = user_data['DecisionScore'].iloc[0]

        # Calculate performance ratio for the rules (needs raw points)
        optimal_points = user_data['OptimalPoints'].iloc[0]
        earned_points = user_data['EarnedPoints'].iloc[0]
        performance_ratio = earned_points / optimal_points

        strategy_text = user_data['StrategyText'].iloc[0]

        # --- RULE SET 1: Psychometrics Override ---
        # Rule: If the user has a low decision score AND is younger (less experience),
        # apply a fixed penalty due to expected difficulty with the task structure.
        if decision_score < 2.5 and age < 30:
            adjustment = 300  # Add 5 minutes (300 seconds)
            final_time += adjustment
            adjustment_reason = f"DecisionScore is very low (<2.5) for younger user (<30). Added {adjustment}s penalty."

        # --- RULE SET 2: Performance Flagging ---
        # Rule: If predicted time is very high AND historical performance is low,
        # flag the task as highly problematic. This is a flag, not a time adjustment.
        elif ml_prediction_time > 2500 and performance_ratio < 0.5:
            adjustment_reason = f"ALERT: ML time is high (>2500s) and historical performance is poor (<50%). High risk of abandonment."

        # --- RULE SET 3: Strategy Text Override (Confidence) ---
        # Rule: If the user's strategy mentions "random" or "no strategy",
        # the ML prediction might be unreliable. Over-adjust towards the median time.
        elif isinstance(strategy_text, str) and (
                'random' in strategy_text.lower() or 'no strategy' in strategy_text.lower()):
            adjustment = 150  # Add a small penalty as uncertainty factor
            final_time += adjustment
            adjustment_reason = f"Strategy indicates 'random' approach. Added {adjustment}s uncertainty penalty."

        return max(500, final_time), adjustment_reason  # Ensure time is realistic


# --- MAIN EXECUTION LOGIC ---

def run_hybrid_system():
    """Main function to run the full hybrid prediction system pipeline."""

    print("--- 1. Data Preparation and Feature Engineering ---")
    data_processor = DataProcessor()
    X, y = data_processor.engineer_features()

    # Example input data to predict on (we will grab one synthetic row)
    # This simulates a new user or a new task set for an existing user.
    sample_data_unprocessed = data_processor.df.iloc[[random.randint(0, len(data_processor.df) - 1)]].drop(
        'TaskCompletionTime', axis=1)

    # --- 2. Train Machine Learning Model ---
    print("\n--- 2. Training ML Model (Random Forest) ---")
    # Pass the data processor instance to the MLPredictor
    ml_predictor = MLPredictor(X, y, data_processor.preprocessor, data_processor)
    ml_predictor.train()

    # --- 3. Run Prediction on Sample Data ---
    print("\n--- 3. Running Prediction on Sample User ---")

    # Pass the UNPROCESSED data to the predict method
    ml_prediction_time = ml_predictor.predict(sample_data_unprocessed)

    print(f"User ID: {sample_data_unprocessed['ID'].iloc[0]}")
    print(f"Raw ML Predicted Completion Time: {ml_prediction_time:.2f} seconds ({ml_prediction_time / 60:.2f} minutes)")

    # --- 4. Apply Rule-Based Adjustment (Hybrid Component) ---
    print("\n--- 4. Applying Hybrid Rule-Based Adjustment ---")
    rule_adjuster = RuleBasedAdjuster()

    # Pass the raw prediction and the *unprocessed* data (including text/scores)
    final_prediction_time, adjustment_reason = rule_adjuster.apply_rules(
        ml_prediction_time,
        sample_data_unprocessed
    )

    # --- 5. Output Results ---
    print("\n--- 5. Final Hybrid System Output ---")
    print(f"ML Prediction: {ml_prediction_time:.2f} seconds")
    print(f"Adjustment Logic: {adjustment_reason}")
    print(f"Final Smart Prediction: {final_prediction_time:.2f} seconds ({final_prediction_time / 60:.2f} minutes)")

    # Show key input features for context
    print("\nKey User Input Features:")
    print(f" - Age: {sample_data_unprocessed['Age'].iloc[0]}")
    print(f" - Decision Score: {sample_data_unprocessed['DecisionScore'].iloc[0]:.2f}")
    print(f" - Strategy Text: '{sample_data_unprocessed['StrategyText'].iloc[0]}'")
    print(f" - Optimal Points: {sample_data_unprocessed['OptimalPoints'].iloc[0]}")


if __name__ == '__main__':
    # Add a check for the VADER library (part of nltk, often needs one-time download)
    try:
        DataProcessor()  # Initialize to check for VADER dependencies
    except LookupError:
        print("\nFATAL ERROR: VADER Lexicon not downloaded. Run the following command once:")
        print("import nltk; nltk.download('vader_lexicon')")
        sys.exit(1)

    run_hybrid_system()
