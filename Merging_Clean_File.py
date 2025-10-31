import pandas as pd
import glob

# List of the uploaded files
file_names = [
    'Metadata.csv',
    'Demographic _102.csv',
    'Psychometrics.csv',
    'Strategy.csv',
    'TaskOrder.csv',
    'Feedback_102.csv'
]


# Function to load a CSV file and clean the 'ID' column
def load_and_clean(file_path):
    df = pd.read_csv(file_path, dtype={'ID': str})
    # Due to some older Excel or data entry issues, some IDs might still be numeric and need conversion
    try:
        df['ID'] = df['ID'].apply(
            lambda x: str(int(float(x))) if pd.notna(x) and isinstance(x, (int, float)) else str(x).split('.')[0]
        )
    except:
        # string conversion
        df['ID'] = df['ID'].astype(str).str.split('.').str[0]

    # Remove whitespace from the IDs
    df['ID'] = df['ID'].str.strip()
    return df

# Creating a dictionary for all cleaned dataframes
dataframes = {}
for file in file_names:
    # Creating a clean, short name for the dataframe (e.g., 'Metadata.csv' -> 'df_meta')
    df_name = 'df_' + file.split('.')[0].replace(' ', '_').lower()
    dataframes[df_name] = load_and_clean(file)
    print(f"Loaded and Cleaned: {file} -> {df_name} (Shape: {dataframes[df_name].shape})")

# Assign individual dataframes for clarity in the merge step
df_meta = dataframes['df_metadata']
df_demo = dataframes['df_demographic__102']
df_psycho = dataframes['df_psychometrics']
df_strategy = dataframes['df_strategy']
df_task = dataframes['df_taskorder']
df_feedback = dataframes['df_feedback_102']

# Starting the merge with the large central dataset (df_meta)
df_merged = df_meta.copy()

# List of remaining dataframes to merge
dfs_to_merge = [df_demo, df_psycho, df_strategy, df_task, df_feedback]

# Performing an inner merge
for i, df_right in enumerate(dfs_to_merge):
    initial_shape = df_merged.shape

    df_merged = pd.merge(
        df_merged,
        df_right,
        on='ID',
        how='inner',
        suffixes=('', f'_{i + 1}')  # Prevents accidental duplicate column names
    )

    print(f"\nMerge step {i + 1}:")
    print(f"  Merged with {df_right.columns.tolist()[1]} data")
    print(f"  Initial Rows: {initial_shape[0]}")
    print(f"  Final Rows: {df_merged.shape[0]}")

# Displaying the results
print("\n" + "=" * 50)
print("✅ Inner Merge Complete!")
print(f"Final Merged DataFrame (df_merged) has a shape of: {df_merged.shape}")
print(f"This means there are **{df_merged.shape[0]}** participants/records with a matching 'ID' in **all six** of your files.")
print("The first few rows of the result:")
print(df_merged.head())