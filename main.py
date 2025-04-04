import streamlit as st 
from datetime import date
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go


START="2010-01-01"
TODAY=date.today().strftime("%Y-%m-%d")

st.title("Stock Forcasting App")

stock=("ONGC.NS","RELIANCE.NS","ADANIENT.NS","^NSEI")
selected_stock=st.selectbox("Select dataset for prediction",stock)

n_years=st.slider("ears of prediction:", 1 , 4)
period=n_years* 365


@st.cache_data
def load_data(ticker):
    data=yf.download(ticker, START,TODAY)
    data.reset_index(inplace=True)  
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce') 
    data.sort_values('Date', inplace=True) 
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col) if col[0] not in ['Date'] else col[0] for col in data.columns.to_flat_index()]  
    

    
    return data

data_load_sate=st.text("Load data...")
data=load_data(selected_stock)
data_load_sate.text("Loading data .... done")

st.subheader('Raw data')
st.write(data.tail())
date_col = 'Date_' if 'Date_' in data.columns else 'Date'
open_col = f"Open_{selected_stock}" if f"Open_{selected_stock}" in data.columns else "Open"
close_col = f"Close_{selected_stock}" if f"Close_{selected_stock}" in data.columns else "Close"
def plot_raw_data():
    # Use the global `data` and `selected_stock`
    date_col = 'Date_' if 'Date_' in data.columns else 'Date'
    open_col = f"Open_{selected_stock}" if f"Open_{selected_stock}" in data.columns else "Open"
    close_col = f"Close_{selected_stock}" if f"Close_{selected_stock}" in data.columns else "Close"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[date_col], y=data[open_col], name='stock_open', mode='lines', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data[date_col], y=data[close_col], name='stock_close', mode='lines', line=dict(color='red'), fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)'))
    fig.update_layout(title_text=f"Time Series Data for {selected_stock}", xaxis=dict(rangeslider=dict(visible=True)))

    st.plotly_chart(fig)
plot_raw_data()


#forecasting 
df_train=data[[date_col,close_col]]
df_train=df_train.rename(columns={date_col:"ds",close_col:"y"})

m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)
st.subheader('Forecast data')
st.write(forecast.tail())

st.write("forecast Data")
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write("forecast component")
fig2=m.plot_components(forecast)
st.write(fig2)