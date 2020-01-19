#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 21:55:43 2020

@author: m
"""
import datetime
from plotly.offline import plot
import plotly.graph_objs as go
from scipy import optimize

import numpy as np
import pandas as pd

import xccy,data
from xccy.data import ProductData, Product, initialize_data
from xccy.modelling import MIN_DATA_DATE
from xccy.bounds import BollingerBand
from xccy.data import PAY, RECEIVE


initialize_data("data")

def get_scores(side, ccy, term):
    pass

date = max(xccy.data.global_data.get_time_series())
side = PAY
ccy = 'JPY'
term = '1Y3Y'
product = Product.from_string(f'{ccy}_{term}')

[k for k in itertools.product(range(1,3),range(1,3))]

lb = 60
# sd = 1
threshold = 0.5
#ma_season_split = 0.75


pdata = ProductData(product, min_date=date, max_date=date)
pdata.closest_fwd(1)
    
season_series = ProductData(product, min_date=MIN_DATA_DATE).series

ma = BollingerBand(lb, lb, sd).get_bounds_df(ma_series).loc[date].to_dict()



lb = 60
sd = 1
threshold = 0.5
ma_season_split = 0.75

# Season score


season_df["season_score"] = season_df["change"].apply(lambda x: (x - season_df["change"].min())/ (season_df["change"].max() - season_df["change"].min()))

def season_score(series):
    season_df = pd.DataFrame({"series": series, 
                             "month": series.index.map(lambda x: x.month),
                             "year": series.index.map(lambda x: x.year)}, index=series.index)
    m_season_df = season_df.groupby(["year", "month"]).mean().reset_index()
    m_season_df["change"] = m_season_df["series"].shift(-1) -  m_season_df["series"]
    m_season_df = m_season_df.groupby("month").mean()[["change"]].reset_index()
    def score(x):
        max_ = max(m_season_df["change"].max(), 1e-5)
        return max(x / max_, 0)
    m_season_df["season_score"] = m_season_df["change"].map(score)
    season_df["index"] = season_df.index
    season_df = season_df.merge(m_season_df, on="month")
    season_df.index = season_df["index"]
    return season_df["season_score"]
 
    

# MA score
def ma_score(b):
    xp = np.array([0, 1.0, 1.50, 2])
    yp = np.array([0, 0.0, 0.25, 1])
    sd = -(b - 0.5) * 2
    return sd.map(lambda x: np.interp(x, xp, yp))

df = BollingerBand(lb, lb, sd).get_bounds_df(series)
df["series"] = series
df["ma_score"] = ma_score(df["%b"])
df["season_score"] = season_score(df["series"])
df["scores"] = df["ma_score"] * ma_season_split + df["season_score"]  * (1 - ma_season_split)
trade_dates = df["scores"][df["scores"] > threshold].index




# viz
title = pdata.product.to_string()
series_plt = go.Scatter(
    x=df.index,
    y=series,
    name = series.name,
    line = dict(color = '#17BECF'),
    opacity = 1)
    
ma_plt = go.Scatter(
    x=df.index,
    y=df["ma"],
    name = str(lb)+"ma",
    line = dict(color = '#7F7F7F'),
    opacity = 0.1)
    
upper_plt = go.Scatter(
    x=df.index,
    y=df["upper"],
    name = "upper",
    line = dict(color = '#7F7F7F'),
    opacity = 0.2)
    
lower_plt = go.Scatter(
    x=df.index,
    y=df["lower"],
    name = "lower",
    line = dict(color = '#7F7F7F'),
    opacity = 0.2)

scores_plt = go.Scatter(
    x=df.index,
    y=df["scores"],
    name = "scores",
    line = dict(color = '#D13636'),
    opacity = 0.2,
    yaxis='y2')
    
trades_markers = go.Scatter(
    x=trade_dates,
    y=series[trade_dates],
    name = "trades",
    mode='markers',
    marker=dict(color='red'),
    opacity = 0.8)

data = [series_plt, ma_plt, upper_plt, lower_plt, scores_plt, trades_markers]
    
layout = dict(
    title=title,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    ),
    yaxis=dict(
        title='yaxis title',
        zeroline=False,
    ),
    yaxis2=dict(
        title='yaxis2 title',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        zeroline=False,
        overlaying='y',
        side='right'
    ),
    #shapes=[make_line_from_ts(ts) for ts in trades.index]
)
fig = dict(data=data, layout=layout)
plot(fig)