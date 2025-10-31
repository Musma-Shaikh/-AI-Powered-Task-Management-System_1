import pandas as pd
import numpy as np
import io
import base64
import sys
import matplotlib

# Set the backend to 'Agg' for non-interactive plotting (required for server environments)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Check for required external libraries (Flask, Prophet, statsmodels)
try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    from prophet import Prophet
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except ImportError as e:
    # Print error and exit if critical libraries are missing
    print(f"FATAL ERROR: Missing required Python package. Please install them:")
    print(f"Error: {e}")
    print("Run: pip install flask pandas numpy matplotlib prophet statsmodels scikit-learn")
    sys.exit(1)

# --- INITIAL SETUP ---
app = Flask(__name__)
# Enable CORS for the frontend running in the Canvas environment
CORS(app)


# --- DATA LOADING AND PREPARATION ---

def prepare_data(df):
    """
    Loads, cleans, and prepares the time series data for analysis.
    The time series variable is the total time spent (in minutes) on tasks per day.
    """
    try:
        # Convert to datetime objects. Added error handling for date parsing.
        try:
            df['StartTimeToronto'] = pd.to_datetime(df['StartTimeToronto'], format='%d-%m-%Y %H:%M', errors='coerce')
            df['EndTimeToronto'] = pd.to_datetime(df['EndTimeToronto'], format='%d-%m-%Y %H:%M', errors='coerce')
        except Exception as date_e:
            print(f"Error parsing dates with format '%d-%m-%Y %H:%M'. Attempting common alternative...")
            # Attempt a more flexible parse if the explicit format fails
            df['StartTimeToronto'] = pd.to_datetime(df['StartTimeToronto'], errors='coerce')
            df['EndTimeToronto'] = pd.to_datetime(df['EndTimeToronto'], errors='coerce')

        # Drop rows where date conversion failed
        df.dropna(subset=['StartTimeToronto', 'EndTimeToronto'], inplace=True)

        if df.empty:
            raise ValueError("Data frame is empty after dropping invalid date rows.")

        # Calculate time difference in seconds
        df['TaskDuration'] = (df['EndTimeToronto'] - df['StartTimeToronto']).dt.total_seconds()

        # Filter out rows where StartTime >= EndTime or duration is negative/zero
        df = df[df['TaskDuration'] > 0]

        if df.empty:
            raise ValueError("Data frame is empty after filtering out invalid durations.")

        # Convert duration from seconds to minutes
        df['TaskDuration_Min'] = df['TaskDuration'] / 60.0

        # Aggregate the total time (in minutes) spent on tasks per day
        # We use the start date for grouping
        df['Date'] = df['StartTimeToronto'].dt.normalize()

        daily_completion_time = df.groupby('Date')['TaskDuration_Min'].sum().reset_index()

        # Prophet requires columns named 'ds' (date) and 'y' (value)
        daily_completion_time.rename(columns={'Date': 'ds', 'TaskDuration_Min': 'y'}, inplace=True)

        # Ensure all dates in the range are present (Prophet likes a complete series)
        date_range = pd.date_range(start=daily_completion_time['ds'].min(),
                                   end=daily_completion_time['ds'].max())
        full_df = pd.DataFrame({'ds': date_range})
        daily_completion_time = pd.merge(full_df, daily_completion_time, on='ds', how='left').fillna(0)

        if daily_completion_time.empty or daily_completion_time['y'].sum() == 0:
            raise ValueError("Data frame is empty or 'y' values are all zero after processing.")

        return daily_completion_time

    except Exception as e:
        print(f"Error during data preparation: {e}")
        # Return an empty list or specific error indicator if data processing fails
        return None


# Load the data globally once at startup
try:
    # Use the specific file name from the uploaded files list
    METADATA_CSV_PATH = 'Metadata.csv'
    print(f"Attempting to read CSV: {METADATA_CSV_PATH}")
    # Assume the user's uploaded CSV is accessible here
    raw_metadata_df = pd.read_csv(METADATA_CSV_PATH)
    data_df = prepare_data(raw_metadata_df)

    if data_df is None:
        raise Exception("Initial data loading and preparation failed (prepare_data returned None).")

except FileNotFoundError:
    print(f"FATAL ERROR: CSV file not found at {METADATA_CSV_PATH}.")
    print("Please ensure 'Metadata.csv' is accessible by the server.")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR during initial data processing: {e}")
    # This exit is what likely caused the 404 because the server never started.
    sys.exit(1)


# --- PLOTTING UTILITIES ---

def plot_to_base64(fig):
    """Converts a Matplotlib figure to a Base64 encoded PNG string."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error during plot conversion to base64: {e}")
        plt.close(fig)
        return ""


def generate_acf_pacf_plots(series):
    """Generates ACF and PACF plots."""
    try:
        # Check if series is long enough for lags
        if len(series) < 2:
            raise ValueError("Time series is too short to generate ACF/PACF plots.")

        lags = min(20, len(series) // 2 - 1)
        if lags <= 0:
            raise ValueError("Not enough data points for meaningful ACF/PACF lags.")

        # Create a subplot for ACF and PACF
        fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
        plot_acf(series, lags=lags, ax=ax_acf, title='Autocorrelation Function (ACF)')
        acf_base64 = plot_to_base64(fig_acf)

        fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4))
        plot_pacf(series, lags=lags, ax=ax_pacf,
                  title='Partial Autocorrelation Function (PACF)')
        pacf_base64 = plot_to_base64(fig_pacf)

        return acf_base64, pacf_base64
    except Exception as e:
        print(f"Error generating ACF/PACF plots: {e}")
        # Return blank plots on failure
        return "", ""


def run_adf_test(series):
    """Performs the Augmented Dickey-Fuller (ADF) test for stationarity."""
    try:
        if len(series) < 5:
            return "ADF Test Error: Need at least 5 data points to run test."

        result = adfuller(series, autolag='AIC')
        output = [
            f"ADF Statistic:        {result[0]:.4f}",
            f"P-value:              {result[1]:.4e}",
            f"Lags Used:            {result[2]}",
            f"Observations Used:    {result[3]}",
            "Critical Values:",
        ]
        for key, value in result[4].items():
            output.append(f"    {key}: {value:.4f}")

        # Conclusion on stationarity
        if result[1] <= 0.05:
            output.append("\nConclusion: The series IS stationary (Reject Null Hypothesis).")
        else:
            output.append("\nConclusion: The series is NOT stationary (Fail to Reject Null Hypothesis).")

        return "\n".join(output)
    except Exception as e:
        return f"ADF Test Error: Could not run test.\n{e}"


# --- FLASK ENDPOINT ---

@app.route('/api/forecast', methods=['GET'])
def forecast_endpoint():
    """
    Endpoint to receive forecast days, run Prophet and ARIMA diagnostics,
    and return all results as JSON.
    """
    if data_df is None or data_df.empty or data_df['y'].sum() == 0:
        return jsonify({
            'status': 'error',
            'message': 'Data is invalid or empty. Check server logs for initial data processing errors.'
        }), 500

    try:
        # 1. Get query parameters
        days_str = request.args.get('days', '30')
        forecast_days = int(days_str)

        if not 1 <= forecast_days <= 90:
            return jsonify({
                'status': 'error',
                'message': 'Forecast days must be between 1 and 90.'
            }), 400

        # --- 2. PROPHET FORECASTING ---
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            interval_width=0.95
        )
        # Fit the model to the data (this can take a few seconds)
        m.fit(data_df)

        # Make future dataframe
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)

        # Plot the main forecast
        fig_prophet = m.plot(forecast)
        fig_prophet.suptitle(f"Prophet Forecast of Daily Task Completion Time ({forecast_days} Days)", y=1.02)
        prophet_plot_base64 = plot_to_base64(fig_prophet)

        # Plot components (Trend, Yearly, Weekly)
        fig_components = m.plot_components(forecast)
        fig_components.suptitle("Prophet Trend and Seasonality Components", y=1.02)
        components_plot_base64 = plot_to_base64(fig_components)

        # --- 3. ARIMA DIAGNOSTICS ---

        # ADF Test
        adf_results = run_adf_test(data_df['y'].values)

        # ACF/PACF Plots (using the original series for diagnostics)
        acf_base64, pacf_base64 = generate_acf_pacf_plots(data_df['y'].values)

        # --- 4. RETURN RESULTS ---

        return jsonify({
            'status': 'success',
            'forecast_days': forecast_days,
            'prophet_plot': prophet_plot_base64,
            'components_plot': components_plot_base64,
            'adf_results': adf_results,
            'acf_plot': acf_base64,
            'pacf_plot': pacf_base64
        })

    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        return jsonify({
            'status': 'error',
            'message': f"An unexpected error occurred during model fitting/prediction: {e}"
        }), 500


# --- RUNNING THE SERVER ---

if __name__ == '__main__':
    print("\n--- FLASK SERVER STARTUP ---")
    print(f"Attempting to load data from: {METADATA_CSV_PATH}")
    if data_df is not None:
        print(f"Data loaded successfully. Total records: {len(data_df)}")
        # Using 8080 as fixed in the previous step
        print("Server running at http://127.0.0.1:8080/api/forecast")
        print("----------------------------")
        # Run the Flask app on the commonly required port 8080
        app.run(host='127.0.0.1', port=8080, debug=False, use_reloader=False)
    else:
        print("Server NOT started due to critical data loading error.")
        # If data fails to load, the server must not start.
        sys.exit(1)
