#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:30:56 2019

@author: m
"""

from plotly.offline import plot

import plotly.graph_objs as go

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