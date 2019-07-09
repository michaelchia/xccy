#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:30:56 2019

@author: m
"""
import datetime
from plotly.offline import plot
import plotly.graph_objs as go

from xccy.data import ProductData
from xccy.modelling import Scorer

MAX_DATE = datetime.datetime(2099, 1, 1)


def plot_eval(model, min_score=1, start_date=None):
    if not start_date:
        start_date = model.date_split_
    pdata = ProductData(model.product, min_date=start_date)
    series = pdata.series
    pred = model.predict(dates=(start_date, MAX_DATE))
    pred = pred[series.index]
    trade_dates = Scorer(min_score).trades(pred, pred).index
    closest_fwd = pdata.closest_fwd()
    _plot(model.product.to_string(ccy=True), 
          series,
          trade_dates,
          pred,
          closest_fwd)


def _plot(title,
          series, 
          trade_dates,
          scores,
          closest_fwd):
    series_plt = go.Scatter(
        x=series.index,
        y=series,
        name = series.name,
        line = dict(color = '#17BECF'),
        opacity = 1)
    
    scores_plt = go.Scatter(
        x=scores.index,
        y=scores,
        name = "scores",
        line = dict(color = '#7F7F7F'),
        opacity = 0.2,
        yaxis='y2')
        
    trades_markers = go.Scatter(
        x=trade_dates,
        y=series[trade_dates],
        name = "trades",
        mode='markers',
        marker=dict(color='red'),
        opacity = 0.8)
    
    closest_fwd_plt = go.Scatter(
        x=closest_fwd.index,
        y=closest_fwd,
        name = closest_fwd.name,
        line = dict(color = '#17BECF'),
        opacity = 0.5)
    
    data = [series_plt, scores_plt, trades_markers, closest_fwd_plt]
        
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
    

def plot_ts(cv, trades, title=None):
    series = go.Scatter(
        x=cv.index,
        y=cv['series'],
        name = "series",
        line = dict(color = '#17BECF'),
        opacity = 0.8)
    
    pred = go.Scatter(
        x=cv.index,
        y=cv['y_pred'],
        name = "pred",
        line = dict(color = '#7F7F7F'),
        opacity = 0.2,
        yaxis='y2')
        
    trades_markers = go.Scatter(
        x=trades.index,
        y=cv['series'][trades.index],
        name = "trades",
        mode='markers',
        marker=dict(color='red'),
        opacity = 0.8)
    
    data = [series, pred, trades_markers]
        
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