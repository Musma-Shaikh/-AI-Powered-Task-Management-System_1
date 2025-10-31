import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from statsmodels.tsa.stattools import adfuller, acf, pacf
# --- NEW: Import CORS from flask_cors ---
from flask_cors import CORS

# NOTE: The 'Prophet' library and its dependencies (like pystan) would need to be installed.
# from prophet import Prophet
# For demonstration, we will use a mock Prophet class.

app = Flask(__name__)
# --- NEW: Enable CORS for all routes/origins ---
CORS(app)


# --- Helper Functions for Data Analysis (Mocks/Placeholders) ---

def load_and_preprocess_data(file_path="Metadata.csv"):
    """
    Loads the user's Metadata.csv file, aggregates task completion times (or counts),
    and prepares it for time series analysis (ds, y format).
    """
    # Placeholder: In a real scenario, you'd load the file and preprocess the data.
    # df = pd.read_csv(file_path)
    # df['ds'] = pd.to_datetime(df['EndTimeToronto']).dt.date
    # time_series_df = df.groupby('ds').size().reset_index(name='y')

    # Generate mock data for demonstration
    date_rng = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
    df_mock = pd.DataFrame(date_rng, columns=['ds'])
    # Add a simple trend + seasonality + noise
    df_mock['y'] = 100 + 0.5 * np.arange(len(df_mock)) + \
                   20 * np.sin(np.linspace(0, 4 * np.pi, len(df_mock))) + \
                   np.random.normal(0, 5, len(df_mock))
    df_mock['y'] = df_mock['y'].round().astype(int)

    # Rename for Prophet compatibility
    time_series_df = df_mock.rename(columns={'ds': 'ds', 'y': 'y'})
    return time_series_df


def generate_plot_base64(fig):
    """Saves a matplotlib figure to a base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_arima_plots(series, d_order=1):
    """
    Generates ACF and PACF plots for ARIMA diagnostics and runs ADF test.

    Args:
        series (pd.Series): The time series data (the 'y' column).
        d_order (int): The differencing order to apply before plotting.

    Returns:
        tuple: (acf_base64, pacf_base64, adf_output_dict)
    """
    try:
        # Run ADF Test on the original series
        adf_test = adfuller(series, autolag='AIC')
        adf_output = {
            'Test Statistic': adf_test[0],
            'p-value': adf_test[1],
            'Lags Used': adf_test[2],
            'Observations Used': adf_test[3],
            'Critical Values': adf_test[4],
            'Result': 'STATIONARY' if adf_test[1] <= 0.05 else 'NON-STATIONARY (Differencing may be needed)'
        }
    except Exception as e:
        adf_output = {'Error': str(e), 'Result': 'ADF Test Failed'}

    # Apply differencing
    differenced_series = series.diff().dropna()

    # ACF Plot
    fig_acf, ax_acf = plt.subplots(figsize=(10, 3))
    _acf_vals = acf(differenced_series, nlags=20, fft=False)
    ax_acf.bar(range(len(_acf_vals)), _acf_vals)
    ax_acf.axhline(y=0, color='gray', linestyle='-')
    ax_acf.axhline(y=-1.96 / np.sqrt(len(differenced_series)), color='r', linestyle='--')
    ax_acf.axhline(y=1.96 / np.sqrt(len(differenced_series)), color='r', linestyle='--')
    ax_acf.set_title(f'Autocorrelation Function (ACF) - Differencing Order d={d_order}')
    acf_base64 = generate_plot_base64(fig_acf)

    # PACF Plot
    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 3))
    _pacf_vals = pacf(differenced_series, nlags=20, method='ols')
    ax_pacf.bar(range(len(_pacf_vals)), _pacf_vals)
    ax_pacf.axhline(y=0, color='gray', linestyle='-')
    ax_pacf.axhline(y=-1.96 / np.sqrt(len(differenced_series)), color='r', linestyle='--')
    ax_pacf.axhline(y=1.96 / np.sqrt(len(differenced_series)), color='r', linestyle='--')
    ax_pacf.set_title(f'Partial Autocorrelation Function (PACF) - Differencing Order d={d_order}')
    pacf_base64 = generate_plot_base64(fig_pacf)

    return acf_base64, pacf_base64, adf_output


# Mock Prophet Class since the actual library is not available in the environment
class MockProphet:
    """Mock class to simulate Prophet's required methods for demonstration."""

    def __init__(self, daily_seasonality=False, weekly_seasonality=False):
        pass

    def fit(self, df):
        print("MockProphet: Fitting mock model.")

    def make_future_dataframe(self, periods, freq='D', include_history=True):
        last_date = pd.to_datetime(ts_data['ds']).max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq=freq)
        history_dates = pd.to_datetime(ts_data['ds'])

        if include_history:
            return pd.DataFrame({'ds': pd.concat([history_dates, pd.Series(future_dates)])})
        else:
            return pd.DataFrame({'ds': future_dates})

    def predict(self, future_df):
        # Generate mock forecast data
        forecast_df = future_df.copy()
        forecast_df['yhat'] = 120 + 0.6 * np.arange(len(forecast_df)) + \
                              15 * np.cos(np.linspace(0, 5 * np.pi, len(forecast_df)))
        forecast_df['yhat_lower'] = forecast_df['yhat'] - 10
        forecast_df['yhat_upper'] = forecast_df['yhat'] + 10
        print("MockProphet: Generated mock prediction.")
        return forecast_df


# Global/Mock data loading
ts_data = load_and_preprocess_data()


# Mock function to simulate Prophet plotting (since fig.add_artist is not available)
def mock_plot(m, forecast, ts_data):
    """Simulates Prophet's plot functionality."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 1. Plot historical data (Prophet uses black dots)
    ax.scatter(ts_data['ds'], ts_data['y'], color='black', label='Historical Data', s=10)

    # 2. Plot forecast uncertainty (Prophet uses blue fill)
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                    color='#0072B2', alpha=0.2, label='Confidence Interval')

    # 3. Plot the forecast trend (Prophet uses blue line)
    ax.plot(forecast['ds'], forecast['yhat'], color='#0072B2', label='Forecasted Task Count')

    # Add a visual split between historical and forecast
    last_hist_date = ts_data['ds'].max()
    ax.axvline(x=last_hist_date, color='red', linestyle='--', alpha=0.7, label='Forecast Start')

    # Cosmetics
    ax.set_xlabel('Date')
    ax.set_ylabel('Task Count')
    ax.set_title('Prophet Forecast (Mock)')
    ax.grid(True, linestyle=':', alpha=0.6)

    return fig


# --- API Route ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get parameters
        # In a real scenario, you'd process the uploaded file here (request.files['file'])
        # Since we use mock data, we only need the 'days' parameter from the URL query or form data
        days_str = request.args.get('days', 30) or request.form.get('days', 30)

        try:
            days = int(days_str)
        except ValueError:
            return jsonify({'status': 'error', 'error': 'Invalid number of days provided.'}), 400

        # 2. Model Initialization and Prediction (Using Mock)
        m = MockProphet(daily_seasonality=True, weekly_seasonality=True)
        m.fit(ts_data)

        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)

        # 3. Generate Prophet Forecast Plot (Fig 1)
        # In a real app, this would be: fig1 = m.plot(forecast)
        fig1 = mock_plot(m, forecast, ts_data)
        fig1.suptitle(f'Prophet Forecast for {days} Days')
        prophet_plot_base64 = generate_plot_base64(fig1)

        # 4. Generate Prophet Components Plot (Fig 2)
        # In a real app, this would be: fig2 = m.plot_components(forecast)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.text(0.5, 0.5, 'Prophet Components Plot Placeholder', ha='center', va='center', fontsize=14)
        ax2.set_title('Prophet Trend & Seasonality Components')
        components_plot_base64 = generate_plot_base64(fig2)

        # --- ARIMA Diagnostics ---
        # We will assume a 'd' (differencing) order of 1 for the purpose of generating the ACF/PACF plots
        acf_base64, pacf_base64, adf_output = generate_arima_plots(ts_data['y'], d_order=1)

        # 5. Return the data
        return jsonify({
            'status': 'success',
            'prophet_plot': prophet_plot_base64,
            'components_plot': components_plot_base64,
            'acf_plot': acf_base64,
            'pacf_plot': pacf_base64,
            'adf_results': adf_output,
            'forecast_days': days
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
