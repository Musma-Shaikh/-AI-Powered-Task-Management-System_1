import pandas as pd

# Load the merged data
df = pd.read_csv("Finale_Merged_Data_csv")

# columns are in datetime format
df['StartTimeToronto'] = pd.to_datetime(df['StartTimeToronto'])
df['EndTimeToronto'] = pd.to_datetime(df['EndTimeToronto'])

# FEATURE ENGINEERING: Calculate Duration
# Convert the time difference to total seconds
df['Duration_Seconds'] = (df['EndTimeToronto'] - df['StartTimeToronto']).dt.total_seconds()

# FEATURE ENGINEERING: Extract Cyclical Features
# Extract day of the week (0=Monday, 6=Sunday)
df['DayOfWeek'] = df['StartTimeToronto'].dt.dayofweek

# Extract hour of the day (0-23)
df['HourOfDay_Start'] = df['StartTimeToronto'].dt.hour

print("--- Datetime Feature Engineering Complete ---")
print(df[['StartTimeToronto', 'EndTimeToronto', 'Duration_Seconds', 'DayOfWeek', 'HourOfDay_Start']].head())