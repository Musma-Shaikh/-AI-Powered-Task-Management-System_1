import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- The Merged Data ---
try:
    df = pd.read_csv("Finale_Merged_Data_csv")
    print(f"Original dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: Finale_Merged_Data_csv not found. Please ensure the file is in the current working directory.")
    exit()

# --- Datetime Feature---

print("\n--- Starting Datetime Feature---")

# Convert columns to datetime objects
df['StartTimeToronto'] = pd.to_datetime(df['StartTimeToronto'], errors='coerce')
df['EndTimeToronto'] = pd.to_datetime(df['EndTimeToronto'], errors='coerce')

# Calculating the TARGET VARIABLE: Duration in Seconds
df['Duration_Seconds'] = (df['EndTimeToronto'] - df['StartTimeToronto']).dt.total_seconds()

# Cleaning up rows where duration is missing or negative (data error)
df = df.dropna(subset=['Duration_Seconds'])
df = df[df['Duration_Seconds'] > 0]

# --- FIX START ---
# creation/concatenation, preventing misalignment errors during TF-IDF steps.
df = df.reset_index(drop=True)

# --- FIX END ---
# Extract time context features
df['DayOfWeek'] = df['StartTimeToronto'].dt.dayofweek  # 0=Monday, 6=Sunday
df['HourOfDay_Start'] = df['StartTimeToronto'].dt.hour  # 0-23
df['Month'] = df['StartTimeToronto'].dt.month

print(f"Cleaned dataset shape: {df.shape}")
print(df[['Duration_Seconds', 'DayOfWeek', 'HourOfDay_Start']].head())

# --- 3. NLP Feature Engineering (Feedback) ---

print("\n--- Starting Feedback NLP (Sentiment) ---")

# Clean the feedback column (replacing the missing/NA values with empty string)
df['Feedback'] = df['Feedback'].fillna('').astype(str).str.replace(r'NA|N/A|NA\.', '', regex=True, case=False)

# Calculate Sentiment Score (Polarity: -1.0 to 1.0)
df['Feedback_Sentiment_Score'] = df['Feedback'].apply(
    lambda text: TextBlob(text).sentiment.polarity
)

# Creating Subjectivity Score (0.0=Objective to 1.0=Subjective)
df['Feedback_Subjectivity'] = df['Feedback'].apply(
    lambda text: TextBlob(text).sentiment.subjectivity
)

print(df[['Feedback', 'Feedback_Sentiment_Score', 'Feedback_Subjectivity']].head())

# --- 4. NLP Feature (Strategy TF-IDF) ---

print("\n--- Starting Strategy NLP (TF-IDF) ---")

# Clean the strategy column
df['Strategy'] = df['Strategy'].fillna('').astype(str)

non_empty_strategies = df[df['Strategy'].str.strip().str.len() > 1]['Strategy']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=50, ngram_range=(1, 2))

if non_empty_strategies.empty:
    print("Warning: All 'Strategy' documents are empty or too short. Skipping TF-IDF feature creation.")
    # Create a dummy feature to avoid future concatenation issues
    tfidf_df = pd.DataFrame(
        np.zeros((len(df), 1)),
        index=df.index,
        columns=['Strategy_TFIDF_Dummy']
    )
else:
    # Fit the TF-IDF vectorizer only on non-empty strategies
    tfidf.fit(non_empty_strategies)

    feature_names = tfidf.get_feature_names_out()

    if len(feature_names) == 0:
        print("Warning: TF-IDF resulted in an empty vocabulary after fitting. Creating a single dummy feature.")
        # Fallback for the edge case where fit() succeeds but generates zero features
        tfidf_df = pd.DataFrame(
            np.zeros((len(df), 1)),
            index=df.index,
            columns=['Strategy_TFIDF_Dummy']
        )
    else:
        # Transform the ENTIRE column
        tfidf_matrix = tfidf.transform(df['Strategy'])

        # Create a DataFrame from the TF-IDF matrix
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=df.index,
            columns=[f'Strategy_TFIDF_{w}' for w in feature_names]
        )

# Concatenate the new TF-IDF features
df = pd.concat([df, tfidf_df], axis=1)

print(f"Created {tfidf_df.shape[1]} new TF-IDF features.")
print(df.filter(regex='Strategy_TFIDF_').head())

print("\n--- Starting One-Hot Encoding for Categorical Data ---")

# Fill NA/missing values with a consistent 'Unknown' label
df['Major'] = df['Major'].fillna('Unknown').astype(str)

# Perform One-Hot Encoding
major_ohe = pd.get_dummies(df['Major'], prefix='Major', dummy_na=False)

# Due to a high number of unique Majors, keeping only the top 10 most frequent categories
top_10_majors = df['Major'].value_counts().nlargest(10).index
major_ohe_filtered = major_ohe.loc[:, [f'Major_{m}' for m in top_10_majors if f'Major_{m}' in major_ohe.columns]]

# Since the index was reset earlier.
df = pd.concat([df, major_ohe_filtered], axis=1).drop('Major', axis=1)

print(f"Added {major_ohe_filtered.shape[1]} Major-related features.")
print(df.filter(regex='^Major_').head())

# ---  Final Save ---

# Save the feature dataset to a new file for the next step
df.to_csv("Feature_Engineered_Data.csv", index=False)

print("\n--- STEP 2 COMPLETE ---")
print("New file 'Feature_Engineered_Data.csv' saved successfully.")
print(f"Final shape of the ML-Ready dataset: {df.shape}")
