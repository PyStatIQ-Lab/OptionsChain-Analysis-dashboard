import streamlit as st
import json
import pandas as pd
import plotly.express as px

# Configure page
st.set_page_config(
    page_title="Options Chain Dashboard",
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
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    with open('upstox_data.json', 'r') as file:
        return json.load(file)

data = load_data()
strike_data = data['data']['strategyChainData']['strikeMap']

# Thresholds
threshold_high = 5000000
threshold_low = 1000000

# Dashboard Header
st.markdown("<div class='header'><h1>📊 Options Chain Dashboard</h1></div>", unsafe_allow_html=True)

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    selected_strike = st.selectbox(
        'Select Strike Price', 
        sorted(list(map(float, strike_data.keys()))),
        index=len(strike_data)//2
    )
    st.markdown("---")
    st.markdown("**Threshold Settings**")
    threshold_high = st.number_input("High Volume Threshold", value=5000000)
    threshold_low = st.number_input("Low Volume Threshold", value=1000000)
    st.markdown("---")
    st.markdown("Built with Streamlit")

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='metric-card'><h3>Put-Call Ratio</h3>", unsafe_allow_html=True)
    pcr = strike_data[str(selected_strike)].get('pcr', 0)
    pcr_color = "positive" if pcr < 1 else "negative"
    st.markdown(f"<h2 class='{pcr_color}'>{pcr:.2f}</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'><h3>Total Call OI Change</h3>", unsafe_allow_html=True)
    call_oi_change = strike_data[str(selected_strike)].get('callOptionData', {}).get('marketData', {}).get('oi', 0) - \
                    strike_data[str(selected_strike)].get('callOptionData', {}).get('marketData', {}).get('prevOi', 0)
    call_color = "positive" if call_oi_change > 0 else "negative"
    st.markdown(f"<h2 class='{call_color}'>{call_oi_change:,}</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='metric-card'><h3>Total Put OI Change</h3>", unsafe_allow_html=True)
    put_oi_change = strike_data[str(selected_strike)].get('putOptionData', {}).get('marketData', {}).get('oi', 0) - \
                   strike_data[str(selected_strike)].get('putOptionData', {}).get('marketData', {}).get('prevOi', 0)
    put_color = "positive" if put_oi_change > 0 else "negative"
    st.markdown(f"<h2 class='{put_color}'>{put_oi_change:,}</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Prepare data for the selected strike
strike_info = strike_data[str(selected_strike)]
call_data = strike_info.get('callOptionData', {})
put_data = strike_info.get('putOptionData', {})

# Create DataFrame for display
df = pd.DataFrame({
    'Metric': [
        'LTP', 'Bid Price', 'Bid Qty', 'Ask Price', 'Ask Qty', 
        'Volume', 'OI', 'OI Change', 'IV', 'Delta',
        'Gamma', 'Vega', 'Theta'
    ],
    'Call': [
        call_data.get('marketData', {}).get('ltp', 0),
        call_data.get('marketData', {}).get('bidPrice', 0),
        call_data.get('marketData', {}).get('bidQty', 0),
        call_data.get('marketData', {}).get('askPrice', 0),
        call_data.get('marketData', {}).get('askQty', 0),
        call_data.get('marketData', {}).get('volume', 0),
        call_data.get('marketData', {}).get('oi', 0),
        call_data.get('marketData', {}).get('oi', 0) - call_data.get('marketData', {}).get('prevOi', 0),
        call_data.get('analytics', {}).get('iv', 0),
        call_data.get('analytics', {}).get('delta', 0),
        call_data.get('analytics', {}).get('gamma', 0),
        call_data.get('analytics', {}).get('vega', 0),
        call_data.get('analytics', {}).get('theta', 0)
    ],
    'Put': [
        put_data.get('marketData', {}).get('ltp', 0),
        put_data.get('marketData', {}).get('bidPrice', 0),
        put_data.get('marketData', {}).get('bidQty', 0),
        put_data.get('marketData', {}).get('askPrice', 0),
        put_data.get('marketData', {}).get('askQty', 0),
        put_data.get('marketData', {}).get('volume', 0),
        put_data.get('marketData', {}).get('oi', 0),
        put_data.get('marketData', {}).get('oi', 0) - put_data.get('marketData', {}).get('prevOi', 0),
        put_data.get('analytics', {}).get('iv', 0),
        put_data.get('analytics', {}).get('delta', 0),
        put_data.get('analytics', {}).get('gamma', 0),
        put_data.get('analytics', {}).get('vega', 0),
        put_data.get('analytics', {}).get('theta', 0)
    ]
})

# Format DataFrame
df['Call'] = df['Call'].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)
df['Put'] = df['Put'].apply(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)

# Display data table
st.markdown(f"<h2 style='margin-top: 20px;'>Strike Price: {selected_strike}</h2>", unsafe_allow_html=True)
st.dataframe(
    df.set_index('Metric'),
    use_container_width=True,
    height=600
)

# Visualization section
st.markdown("---")
st.markdown("<h2>Options Flow Analysis</h2>", unsafe_allow_html=True)

# Calculate OI changes for nearby strikes
strikes = sorted(list(map(float, strike_data.keys())))
current_idx = strikes.index(selected_strike)
nearby_strikes = strikes[max(0, current_idx-5):min(len(strikes), current_idx+6)]

call_oi_changes = []
put_oi_changes = []
for strike in nearby_strikes:
    call_oi = strike_data[str(strike)].get('callOptionData', {}).get('marketData', {}).get('oi', 0)
    prev_call_oi = strike_data[str(strike)].get('callOptionData', {}).get('marketData', {}).get('prevOi', 0)
    call_oi_changes.append(call_oi - prev_call_oi)
    
    put_oi = strike_data[str(strike)].get('putOptionData', {}).get('marketData', {}).get('oi', 0)
    prev_put_oi = strike_data[str(strike)].get('putOptionData', {}).get('marketData', {}).get('prevOi', 0)
    put_oi_changes.append(put_oi - prev_put_oi)

# Create OI change plot
fig = px.bar(
    x=nearby_strikes,
    y=[call_oi_changes, put_oi_changes],
    barmode='group',
    labels={'x': 'Strike Price', 'y': 'OI Change'},
    title=f'Open Interest Changes Around {selected_strike}',
    color_discrete_map={0: '#3498db', 1: '#e74c3c'}
)
fig.update_layout(
    legend_title_text='Option Type',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
st.plotly_chart(fig, use_container_width=True)

# Sentiment analysis
st.markdown("---")
st.markdown("<h2>Market Sentiment Analysis</h2>", unsafe_allow_html=True)

sentiment_cols = st.columns(3)
with sentiment_cols[0]:
    st.markdown("<div class='metric-card'><h4>Call Sentiment</h4>", unsafe_allow_html=True)
    call_sentiment = "Bullish" if call_oi_change > 0 else "Bearish"
    sentiment_color = "positive" if call_sentiment == "Bullish" else "negative"
    st.markdown(f"<h3 class='{sentiment_color}'>{call_sentiment}</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with sentiment_cols[1]:
    st.markdown("<div class='metric-card'><h4>Put Sentiment</h4>", unsafe_allow_html=True)
    put_sentiment = "Bullish" if put_oi_change > 0 else "Bearish"
    sentiment_color = "positive" if put_sentiment == "Bullish" else "negative"
    st.markdown(f"<h3 class='{sentiment_color}'>{put_sentiment}</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with sentiment_cols[2]:
    st.markdown("<div class='metric-card'><h4>Overall Sentiment</h4>", unsafe_allow_html=True)
    if pcr < 0.7:
        overall_sentiment = "Strongly Bullish"
        sentiment_color = "positive"
    elif pcr < 1:
        overall_sentiment = "Mildly Bullish"
        sentiment_color = "positive"
    elif pcr < 1.3:
        overall_sentiment = "Neutral"
        sentiment_color = ""
    else:
        overall_sentiment = "Bearish"
        sentiment_color = "negative"
    st.markdown(f"<h3 class='{sentiment_color}'>{overall_sentiment}</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Volume-OI Analysis
st.markdown("---")
st.markdown("<h2>Volume-Open Interest Analysis</h2>", unsafe_allow_html=True)

vol_oi_cols = st.columns(2)
with vol_oi_cols[0]:
    st.markdown("<div class='metric-card'><h4>Call Analysis</h4>", unsafe_allow_html=True)
    call_vol = call_data.get('marketData', {}).get('volume', 0)
    if call_vol > threshold_high and call_oi_change > 0:
        analysis = "Strong Trend: High Volume + Increasing OI"
        color = "positive"
    elif (call_vol > threshold_high and call_oi_change < 0) or (call_vol < threshold_low and call_oi_change > 0):
        analysis = "Potential Reversal: Divergence"
        color = "negative"
    else:
        analysis = "No Clear Signal"
        color = ""
    st.markdown(f"<p class='{color}'>{analysis}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with vol_oi_cols[1]:
    st.markdown("<div class='metric-card'><h4>Put Analysis</h4>", unsafe_allow_html=True)
    put_vol = put_data.get('marketData', {}).get('volume', 0)
    if put_vol > threshold_high and put_oi_change > 0:
        analysis = "Strong Trend: High Volume + Increasing OI"
        color = "positive"
    elif (put_vol > threshold_high and put_oi_change < 0) or (put_vol < threshold_low and put_oi_change > 0):
        analysis = "Potential Reversal: Divergence"
        color = "negative"
    else:
        analysis = "No Clear Signal"
        color = ""
    st.markdown(f"<p class='{color}'>{analysis}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
