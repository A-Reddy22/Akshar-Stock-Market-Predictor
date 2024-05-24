import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Akshar's Stock Prediction App")
stocks = ("TDOC", "PLTR", "MSFT", "SOFI", "SPY","TSLA","AAPL","BABA","GS","PINS", "Z", "SWKS","TOL", "MA","CRUS", "NVDA","SNOW","Sq","PYPL","MMM","MP","DBX")
selected_stocks = st.selectbox("Select data for prediction", stocks)

n_years = st.slider("Years of prediction", 1, 37)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting with Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast Data')
st.write(forecast.tail())

# Plot forecast
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast Components')

# Ensure columns in the forecast DataFrame are not of dtype object
forecast['ds'] = forecast['ds'].astype('datetime64[ns]')
forecast['yhat'] = forecast['yhat'].astype(float)
forecast['yhat_lower'] = forecast['yhat_lower'].astype(float)
forecast['yhat_upper'] = forecast['yhat_upper'].astype(float)

fig2 = plot_components_plotly(m, forecast)
st.plotly_chart(fig2)















