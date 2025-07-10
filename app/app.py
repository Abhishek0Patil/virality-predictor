import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pytrends.request import TrendReq
from pytrends import exceptions
import os
from datetime import datetime, timedelta

# ==============================================================================
# Page Configuration & Path Setup
# ==============================================================================
st.set_page_config(
    page_title="The Virality Predictor",
    page_icon="🔮",
    layout="wide"
)

# --- Define Paths Robustly ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
ASSETS_DIR = os.path.join(SCRIPT_DIR, 'assets')

st.title("🔮 The Virality Predictor")
st.markdown("Analyze the early-stage DNA of a new trend to forecast its future popularity.")

# ==============================================================================
# Helper Functions
# ==============================================================================

@st.cache_resource
def load_model():
    """Loads the trained model from disk using a robust, absolute path."""
    model_path = os.path.join(MODELS_DIR, 'virality_predictor_geo_v2_regularized.pkl')
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"FATAL: Model file not found at {model_path}. The application cannot start.")
        return None

@st.cache_data
def engineer_features(df):
    """Performs the same simplified feature engineering as the V3 training script."""
    if 'gtrends_volume' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'gtrends_volume'})
    df_features = df[['gtrends_volume']].copy()
    for window in [7, 14]:
        df_features[f'vol_rolling_mean_{window}'] = df_features['gtrends_volume'].rolling(window=window).mean()
        df_features[f'vol_rolling_std_{window}'] = df_features['gtrends_volume'].rolling(window=window).std()
    for lag in [1, 7, 14]:
        df_features[f'vol_lag_{lag}'] = df_features['gtrends_volume'].shift(lag)
    df_features['vol_diff_7'] = df_features['gtrends_volume'].diff(7)
    df_features.fillna(0, inplace=True)
    return df_features

@st.cache_data
def fetch_trend_data(keyword, geo, days=90):
    """Fetches Google Trends data. Returns a DataFrame on success or None on failure."""
    pytrends = TrendReq(hl='en-US', tz=360)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
    try:
        pytrends.build_payload(kw_list=[keyword], timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time()
        if 'isPartial' in df.columns: df = df.drop(columns=['isPartial'])
        if df.empty or keyword not in df.columns: return None
        return df
    except Exception as e:
        print(f"Error fetching data for '{keyword}': {e}")
        return None

def generate_prediction_summary(prediction, keyword):
    """Analyzes the prediction array and generates a more nuanced summary."""
    peak_value = np.max(prediction)
    peak_day = np.argmax(prediction) + 1
    avg_value = np.mean(prediction)
    initial_value = prediction[0]
    final_value = prediction[-1]
    
    if peak_value > 50 and peak_day <= 7:
        qualitative_summary = f"The model predicts a significant and immediate peak for '{keyword}', followed by a rapid decline. This suggests a short-lived, high-intensity 'flash in the pan' trend."
    elif peak_value > initial_value * 1.5 and peak_day > 7:
        qualitative_summary = f"The forecast suggests '{keyword}' will grow in popularity, reaching a peak later in the month before declining. This indicates a developing, sustained interest."
    elif final_value < initial_value * 0.5:
        qualitative_summary = f"The forecast shows a steady decay in interest for '{keyword}', suggesting the trend is likely past its peak and will gradually fade."
    else:
        qualitative_summary = f"The model predicts a stable plateau for '{keyword}', suggesting it will maintain its current level of interest over the next month."

    summary_text = f"""
    ### Prediction Insights for '{keyword}'

    - **Peak Popularity:** The trend is expected to reach a peak Google Trends score of **{peak_value:.0f}** approximately **{peak_day} days** from the start of the forecast.
    - **Average Interest:** Over the next 28 days, the average interest level is predicted to be around **{avg_value:.0f}**.
    - **Overall Outlook:** {qualitative_summary}
    """
    return summary_text

# ==============================================================================
# Main App UI and Logic
# ==============================================================================

model = load_model()
col1, col2 = st.columns(2)
with col1:
    keyword = st.text_input("Enter a keyword:", "New movie trailer")
with col2:
    geo = st.selectbox("Select a region:", ["Worldwide", "India"], index=0)
geo_code = "" if geo == "Worldwide" else "IN"

if st.button("🚀 Predict Virality", type="primary"):
    if not keyword:
        st.warning("Please enter a keyword.")
    elif model is None:
        st.error("Model is not loaded.")
    else:
        with st.spinner(f"Analyzing data for '{keyword}'..."):
            history_days = 14
            initial_fetch_days = history_days + 15 
            raw_df = fetch_trend_data(keyword, geo_code, days=initial_fetch_days)

        if raw_df is None or len(raw_df) < history_days:
            st.error(f"Failed to fetch sufficient data for '{keyword}'. This could be due to a temporary rate limit by Google or because the trend is too new/unpopular.")
            st.markdown("---")
            st.subheader("For reference, here's what a classic trend lifecycle looks like:")
            fallback_image_path = os.path.join(ASSETS_DIR, 'fallback_chatgpt.png')
            if os.path.exists(fallback_image_path):
                st.image(fallback_image_path, caption="The 'ChatGPT' trend shows a classic S-curve: a slow start, explosive growth, a peak, and a gradual plateau of high interest.")
            else:
                st.info("Fallback example image not found.")
        else: 
            st.success(f"Data for '{keyword}' fetched successfully! Now running prediction...")
            
            df_history = raw_df.tail(history_days).copy()
            df_engineered = engineer_features(raw_df).tail(history_days)
            
            x_temporal = df_engineered.values.flatten()
            geo_IN = 1 if geo_code == 'IN' else 0
            geo_WW = 1 if geo_code == '' else 0
            x_geo = np.array([geo_IN, geo_WW])
            X_final = np.concatenate([x_temporal, x_geo]).reshape(1, -1)
            
            prediction = model.predict(X_final)[0]

            st.subheader("Forecasted Trend Lifecycle")
            history_dates = df_history.index
            future_dates = pd.to_datetime(pd.date_range(start=history_dates[-1] + timedelta(days=1), periods=28))
            volume_history = df_history[keyword].values if keyword in df_history.columns else df_history.iloc[:, 0].values
            
            chart_df = pd.DataFrame({
                'Date': np.concatenate([history_dates, future_dates]),
                'Volume': np.concatenate([volume_history, prediction]),
                'Type': ['Known History'] * len(history_dates) + ['Predicted Future'] * len(prediction)
            })
            st.line_chart(chart_df, x='Date', y='Volume', color='Type', height=500)
            
            # --- The summary is now the final piece of output ---
            summary = generate_prediction_summary(prediction, keyword)
            st.markdown(summary)
            
            # --- THE DOWNLOAD BUTTON SECTION HAS BEEN COMPLETELY REMOVED ---

st.markdown("---")
st.markdown("Built by Abhishek Patil. This tool uses an XGBoost model trained on historical Google Trends data to forecast future popularity.")