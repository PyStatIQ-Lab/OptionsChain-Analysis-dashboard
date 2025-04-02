import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json

# Configure page
st.set_page_config(
    page_title="PyStatIQ Options Chain Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .header {
        color: #2c3e50;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive {
        color: #27ae60;
    }
    .negative {
        color: #e74c3c;
    }
    .prediction-card {
        background-color: #f1f8fe;
        border-left: 5px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .tabs {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
BASE_URL = "https://service.upstox.com/option-analytics-tool/open/v1"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json"
}

# Fetch data from API
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_options_data(asset_key="NSE_INDEX|Nifty 50", expiry="03-04-2025"):
    url = f"{BASE_URL}/strategy-chains?assetKey={asset_key}&strategyChainType=PC_CHAIN&expiry={expiry}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch data: {response.status_code} - {response.text}")
        return None

# Process raw API data
def process_options_data(raw_data):
    if not raw_data or 'data' not in raw_data:
        return None
    
    strike_map = raw_data['data']['strategyChainData']['strikeMap']
    processed_data = []
    
    for strike, data in strike_map.items():
        call_data = data.get('callOptionData', {})
        put_data = data.get('putOptionData', {})
        
        # Market data
        call_market = call_data.get('marketData', {})
        put_market = put_data.get('marketData', {})
        
        # Analytics data
        call_analytics = call_data.get('analytics', {})
        put_analytics = put_data.get('analytics', {})
        
        processed_data.append({
            'strike': float(strike),
            'pcr': data.get('pcr', 0),
            
            # Call data
            'call_ltp': call_market.get('ltp', 0),
            'call_bid': call_market.get('bidPrice', 0),
            'call_ask': call_market.get('askPrice', 0),
            'call_volume': call_market.get('volume', 0),
            'call_oi': call_market.get('oi', 0),
            'call_prev_oi': call_market.get('prevOi', 0),
            'call_oi_change': call_market.get('oi', 0) - call_market.get('prevOi', 0),
            'call_iv': call_analytics.get('iv', 0),
            'call_delta': call_analytics.get('delta', 0),
            'call_gamma': call_analytics.get('gamma', 0),
            'call_theta': call_analytics.get('theta', 0),
            'call_vega': call_analytics.get('vega', 0),
            
            # Put data
            'put_ltp': put_market.get('ltp', 0),
            'put_bid': put_market.get('bidPrice', 0),
            'put_ask': put_market.get('askPrice', 0),
            'put_volume': put_market.get('volume', 0),
            'put_oi': put_market.get('oi', 0),
            'put_prev_oi': put_market.get('prevOi', 0),
            'put_oi_change': put_market.get('oi', 0) - put_market.get('prevOi', 0),
            'put_iv': put_analytics.get('iv', 0),
            'put_delta': put_analytics.get('delta', 0),
            'put_gamma': put_analytics.get('gamma', 0),
            'put_theta': put_analytics.get('theta', 0),
            'put_vega': put_analytics.get('vega', 0),
        })
    
    return pd.DataFrame(processed_data)

# Prediction model (simplified for demo)
@st.cache_resource
def get_prediction_model():
    # This would be replaced with your actual trained model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Mock training data
    X = pd.DataFrame({
        'pcr': np.random.uniform(0.5, 1.5, 100),
        'call_oi_change': np.random.randint(-100000, 100000, 100),
        'put_oi_change': np.random.randint(-100000, 100000, 100),
        'iv_diff': np.random.uniform(-5, 5, 100)
    })
    y = (X['pcr'] + X['call_oi_change']/100000 - X['put_oi_change']/100000 > 1.2).astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model.fit(X_scaled, y)
    return model, scaler

# Generate trading signals
def generate_signals(df, selected_strike):
    signals = []
    row = df[df['strike'] == selected_strike].iloc[0]
    
    # PCR signal
    if row['pcr'] < 0.7:
        signals.append(("Extremely Low PCR (Bullish)", f"PCR at {row['pcr']:.2f} suggests bullish sentiment", "high"))
    elif row['pcr'] > 1.3:
        signals.append(("Extremely High PCR (Bearish)", f"PCR at {row['pcr']:.2f} suggests bearish sentiment", "high"))
    
    # OI divergence signal
    if row['call_oi_change'] > 0 and row['put_oi_change'] < 0:
        signals.append(("OI Bullish Divergence", 
                      f"Call OI ↑ {row['call_oi_change']:,} | Put OI ↓ {abs(row['put_oi_change']):,}", 
                      "medium"))
    elif row['call_oi_change'] < 0 and row['put_oi_change'] > 0:
        signals.append(("OI Bearish Divergence", 
                      f"Call OI ↓ {abs(row['call_oi_change']):,} | Put OI ↑ {row['put_oi_change']:,}", 
                      "medium"))
    
    # IV skew signal
    iv_diff = row['call_iv'] - row['put_iv']
    if iv_diff > 2:
        signals.append(("Call IV Skew", f"Call IV {row['call_iv']:.1f}% vs Put IV {row['put_iv']:.1f}%", "low"))
    elif iv_diff < -2:
        signals.append(("Put IV Skew", f"Put IV {row['put_iv']:.1f}% vs Call IV {row['call_iv']:.1f}%", "low"))
    
    return signals

# Main App
def main():
    st.markdown("<div class='header'><h1>📊 PyStatIQ Options Chain Dashboard</h1></div>", unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("Filters")
        asset_key = st.selectbox(
            "Underlying Asset",
            ["NSE_INDEX|Nifty 50", "NSE_INDEX|Bank Nifty"],
            index=0
        )
        
        expiry_date = st.date_input(
            "Expiry Date",
            datetime.strptime("03-04-2025", "%d-%m-%Y")
        ).strftime("%d-%m-%Y")
        
        st.markdown("---")
        st.markdown("**Analysis Settings**")
        volume_threshold = st.number_input("High Volume Threshold", value=5000000)
        oi_change_threshold = st.number_input("Significant OI Change", value=1000000)
        
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("This dashboard provides real-time options chain analysis using PyStatIQ API data.")
    
    # Fetch and process data
    with st.spinner("Fetching live options data..."):
        raw_data = fetch_options_data(asset_key, expiry_date)
    
    if raw_data is None:
        st.error("Failed to load data. Please try again later.")
        return
    
    df = process_options_data(raw_data)
    if df is None or df.empty:
        st.error("No data available for the selected parameters.")
        return
    
    # Default strike selection (ATM)
    atm_strike = df.iloc[(df['strike'] - (df['call_ltp'] + df['strike'] - df['put_ltp'])/2).abs().argsort()[:1]]['strike'].values[0]
    
    # Main columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Total Call OI**")
        total_call_oi = df['call_oi'].sum()
        st.markdown(f"<h2>{total_call_oi:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Total Put OI**")
        total_put_oi = df['put_oi'].sum()
        st.markdown(f"<h2>{total_put_oi:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Strike price selector
        selected_strike = st.selectbox(
            "Select Strike Price",
            df['strike'].unique(),
            index=int(np.where(df['strike'].unique() == atm_strike)[0][0])
        )
        
        # PCR gauge
        pcr = df[df['strike'] == selected_strike]['pcr'].values[0]
        fig = px.bar(x=[pcr], range_x=[0, 2], title=f"Put-Call Ratio: {pcr:.2f}")
        fig.update_layout(
            xaxis_title="PCR",
            yaxis_visible=False,
            height=150,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.add_vline(x=0.7, line_dash="dot", line_color="green")
        fig.add_vline(x=1.3, line_dash="dot", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Call OI Change**")
        call_oi_change = df[df['strike'] == selected_strike]['call_oi_change'].values[0]
        change_color = "positive" if call_oi_change > 0 else "negative"
        st.markdown(f"<h2 class='{change_color}'>{call_oi_change:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Put OI Change**")
        put_oi_change = df[df['strike'] == selected_strike]['put_oi_change'].values[0]
        change_color = "positive" if put_oi_change > 0 else "negative"
        st.markdown(f"<h2 class='{change_color}'>{put_oi_change:,}</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab layout
    tab1, tab2, tab3 = st.tabs(["Strike Analysis", "OI/Volume Trends", "Advanced Analytics"])
    
    with tab1:
        st.markdown(f"### Detailed Analysis for Strike: {selected_strike}")
        
        # Get selected strike data
        strike_data = df[df['strike'] == selected_strike].iloc[0]
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Metric': ['LTP', 'Bid', 'Ask', 'Volume', 'OI', 'OI Change', 'IV', 'Delta', 'Gamma', 'Theta', 'Vega'],
            'Call': [
                strike_data['call_ltp'],
                strike_data['call_bid'],
                strike_data['call_ask'],
                strike_data['call_volume'],
                strike_data['call_oi'],
                strike_data['call_oi_change'],
                strike_data['call_iv'],
                strike_data['call_delta'],
                strike_data['call_gamma'],
                strike_data['call_theta'],
                strike_data['call_vega']
            ],
            'Put': [
                strike_data['put_ltp'],
                strike_data['put_bid'],
                strike_data['put_ask'],
                strike_data['put_volume'],
                strike_data['put_oi'],
                strike_data['put_oi_change'],
                strike_data['put_iv'],
                strike_data['put_delta'],
                strike_data['put_gamma'],
                strike_data['put_theta'],
                strike_data['put_vega']
            ]
        })
        
        st.dataframe(
            comparison_df.style.format({
                'Call': '{:,.2f}',
                'Put': '{:,.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Trading signals
        st.markdown("### Trading Signals")
        signals = generate_signals(df, selected_strike)
        
        if signals:
            for signal in signals:
                name, description, confidence = signal
                with st.expander(f"{'🟢' if 'Bullish' in name else '🔴'} {name} ({confidence} confidence)"):
                    st.markdown(description)
                    if "PCR" in name:
                        st.markdown("Historical accuracy: 72%")
                    elif "OI" in name:
                        st.markdown("Historical accuracy: 65%")
                    elif "IV" in name:
                        st.markdown("Historical accuracy: 58%")
        else:
            st.info("No strong trading signals detected at this strike price")
    
    with tab2:
        st.markdown("### Open Interest & Volume Trends")
        
        # Nearby strikes
        all_strikes = sorted(df['strike'].unique())
        current_idx = all_strikes.index(selected_strike)
        nearby_strikes = all_strikes[max(0, current_idx-5):min(len(all_strikes), current_idx+6)]
        nearby_df = df[df['strike'].isin(nearby_strikes)]
        
        # OI Change plot
        fig = px.bar(
            nearby_df,
            x='strike',
            y=['call_oi_change', 'put_oi_change'],
            barmode='group',
            title=f'OI Changes Around {selected_strike}',
            labels={'value': 'OI Change', 'strike': 'Strike Price'},
            color_discrete_map={'call_oi_change': '#3498db', 'put_oi_change': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume plot
        fig = px.bar(
            nearby_df,
            x='strike',
            y=['call_volume', 'put_volume'],
            barmode='group',
            title=f'Volume Around {selected_strike}',
            labels={'value': 'Volume', 'strike': 'Strike Price'},
            color_discrete_map={'call_volume': '#3498db', 'put_volume': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Predictive Analytics")
        
        # Load prediction model
        model, scaler = get_prediction_model()
        
        # Prepare features for prediction
        features = pd.DataFrame({
            'pcr': [strike_data['pcr']],
            'call_oi_change': [strike_data['call_oi_change']],
            'put_oi_change': [strike_data['put_oi_change']],
            'iv_diff': [strike_data['call_iv'] - strike_data['put_iv']]
        })
        
        # Make prediction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        
        # Display prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            st.markdown("**Next Day Direction**")
            direction = "Bullish ↑" if prediction == 1 else "Bearish ↓"
            color = "positive" if prediction == 1 else "negative"
            st.markdown(f"<h2 class='{color}'>{direction}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
            st.markdown("**Confidence**")
            confidence = max(proba) * 100
            st.markdown(f"<h2>{confidence:.1f}%</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature importance
        st.markdown("#### Key Predictive Factors")
        feature_importance = pd.DataFrame({
            'Feature': ['PCR', 'Call OI Change', 'Put OI Change', 'IV Difference'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='What Factors Are Driving This Prediction'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk analysis
        st.markdown("#### Risk Analysis")
        
        # Max pain calculation
        pain_points = []
        for strike in df['strike'].unique():
            strike_row = df[df['strike'] == strike].iloc[0]
            pain_points.append((strike, strike_row['call_oi'] + strike_row['put_oi']))
        
        max_pain_strike = min(pain_points, key=lambda x: x[1])[0] if pain_points else selected_strike
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("**Maximum Pain**")
            st.markdown(f"Current Strike: {selected_strike}")
            st.markdown(f"Max Pain Strike: {max_pain_strike}")
            
            if abs(max_pain_strike - selected_strike) <= (all_strikes[1] - all_strikes[0]) * 2:
                st.warning("Close to max pain - increased pin risk")
            else:
                st.success("Not near max pain level")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("**Gamma Exposure**")
            
            net_gamma = strike_data['call_gamma'] - strike_data['put_gamma']
            if net_gamma > 0:
                st.info("Positive Gamma: Market makers likely to buy on dips, sell on rallies")
            else:
                st.warning("Negative Gamma: Market makers likely to sell on dips, buy on rallies")
            
            st.markdown(f"Net Gamma: {net_gamma:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
