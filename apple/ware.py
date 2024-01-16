import yfinance as yf

# import numpy as np
import pandas as pd

import plotly
import plotly.graph_objs as go
import plotly.express as px



def get_closing_prices(symbol, period="1mo"):  # default value of 1 day.
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period)
        return data[["Close",'Volume']]
    except Exception as e:
        print("Failed to get required data.", e)

ticker = "AAPL"
ticker2 = "BA"
ticker3 = "DDOG"
ticker4 = "ES=F"

period = "20y"


apple = pd.DataFrame(get_closing_prices(ticker, period)).round().\
reset_index().groupby(pd.Grouper(key='Date',freq='W')).agg({'Close':'mean','Volume':'sum'})
boeing = pd.DataFrame(get_closing_prices(ticker2, period)).round().\
reset_index().groupby(pd.Grouper(key='Date',freq='W')).agg({'Close':'mean','Volume':'sum'})
ddog = pd.DataFrame(get_closing_prices(ticker3, period)).round().\
reset_index().groupby(pd.Grouper(key='Date',freq='W')).agg({'Close':'mean','Volume':'sum'})
snp = (pd.DataFrame(get_closing_prices(ticker4, period))/40).round().\
reset_index().groupby(pd.Grouper(key='Date',freq='W')).agg({'Close':'mean','Volume':'sum'})

fig = go.Figure()
fig.add_trace(go.Scatter(x=apple.index, y=apple['Close'], name = 'Apple',
                         mode='lines+markers',
                         marker=dict(color=apple['Volume'])))
fig.add_trace(go.Scatter(x=ddog.index, y=ddog['Close'], name = 'DataDog',
                         mode='lines+markers',
                         marker=dict(color=ddog['Volume'])))
fig.add_trace(go.Scatter(x=boeing.index, y=boeing['Close'], name = 'The Boeing Company',
                         mode='lines+markers',
                         marker=dict(color=boeing['Volume'])))
fig.add_trace(go.Scatter(x=snp.index, y=snp['Close'], name='S&P500'))

fig.update_layout(legend_orientation="h",
                  margin=dict(l=15, r=20, t=30, b=10),
                  legend=dict(x=.5, xanchor="center"),
                  title="American Stocks",
                  hovermode="x",
                  template='seaborn',
                  yaxis_title="$ Price",)
fig.update_traces(hoverinfo="all", hovertemplate="Дата: %{x}<br>Цена: %{y}")

graf = fig