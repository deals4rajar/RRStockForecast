# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from neuralprophet import NeuralProphet
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

import plotly.graph_objects as go


st.title('Stock Forecast App')


stock_symbol = st.text_input('Enter Stock Symnbol', 'GOOG')

if  stock_symbol.find("=")<0:
	stockdesc = yf.Ticker(stock_symbol)
	st.write(stockdesc.info['longName'])

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

start_date = '2015-01-01'
end_date = date.today().strftime("%Y-%m-%d")

stock_data = yf.download(stock_symbol, start = start_date, end=end_date)

print(stock_data.head())
stock_data.to_csv('stock_data.csv')

stocks = pd.DataFrame(stock_data)
stocks.reset_index(inplace=True)
stocks['Date'] = pd.to_datetime(stocks['Date'])
stocks = stocks[['Date', 'Close']]
stocks.columns = ['ds', 'y']

plt.plot(stocks['ds'], stocks['y'], label = 'actual', c = 'g')
plt.show()

st.subheader('Raw data')
st.write(stocks.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=stocks['ds'], y=stocks['y'], name="Actual"))
	fig.layout.update(title_text='Actual Stock Price Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

model = Prophet()
model.fit(stocks)

future = model.make_future_dataframe(periods = period)
future.reset_index(inplace=True)

forecast = model.predict(future)
actual_prediction = model.predict(stocks)

# Show and plot actual_prediction
st.subheader('Actual prediction')
st.write(actual_prediction.tail())

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    

Weekly = model.make_future_dataframe(periods = 10)

Weeklyforecast = model.predict(Weekly)

# Show and plot forecast
st.subheader('Weekly Forecast data')
st.write(Weeklyforecast.tail())

st.subheader(f'Forecast plot for {n_years} years')

#st.write(f'Forecast plot for {n_years} years')
fig5 = plot_plotly(model, forecast)
st.write(fig5)

st.write("Forecast components")
#fig3 = model.plot_components(forecast)
fig3 = plot_components_plotly(model, forecast)
st.write(fig3)

