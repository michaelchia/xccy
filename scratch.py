#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 01:24:23 2019

@author: m
"""
import itertools
from copy import copy
import importlib
from pprint import pprint as print

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import scipy


import xccy
import xccy.data
import xccy.feature_engineering
import xccy.modelling
importlib.reload(xccy)
from xccy import data
from xccy.data import Product, ProductData, filter_df
importlib.reload(xccy.feature_engineering)
from xccy.feature_engineering import FeatEng, RegLabel
importlib.reload(xccy.modelling)
from xccy.modelling import Classifier, make_training_data, Scorer, ProductModel


def get_quarter_pos(date):
    q = int(date.month / 4)
    m_start = q * 3 + 1
    m_end = (q + 1) * 3 + 1
    start = datetime.datetime(date.year, m_start, 1) \
        if m_start <= 12 else datetime.datetime(date.year + 1, m_start - 12, 1)
    end = datetime.datetime(date.year, m_end, 1) \
        if m_end <= 12 else datetime.datetime(date.year + 1, m_end - 12, 1)
    t_days = (end - start).days
    days = (date - start).days
    return days / t_days
    

def plot_df(df, label_col='label'):
    for col in df.columns:
        if col != label_col:
            plt.scatter(df[col], df[label_col], c='k', label='data', s=7)
            print(np.corrcoef(df[col], df[label_col])[0][1])
            
def grid_gen(search_space):
    return ({k:v for k, v in zip(search_space.keys(), params)} 
            for params in itertools.product(*search_space.values()))


data.initialize_data('/Users/m/Documents/xccy2/data')
jpy = data.global_data.local_data('EUR')._src_df
p = Product.from_string('AUD_5Y5Y')


pmodel = ProductModel(p)

cv = pmodel.model.cv_data_
cv['series'] = pdata.product_series()
trades = Scorer(1).trades(cv['y'], cv['y_pred'])
print(Scorer(1).evaluate(cv['y'], cv['y_pred']))

for i in range(25, 225, 25):
    ev = Scorer(i/100).evaluate(cv['y'], cv['y_pred'])
    print('{:2g}: {:2g} {} {:2g}'.format(ev['threshold'], ev['score'], ev['n'], ev['trades']['pos']/ev['trades']['neg']))

plot_ts(cv, trades, title=p.to_string())



import plotly.plotly as py
from plotly.offline import plot

import plotly.graph_objs as go

def plot_ts(cv, trades, title=None):
    trace_high = go.Scatter(
        x=cv.index,
        y=cv['series'],
        name = "series",
        line = dict(color = '#17BECF'),
        opacity = 0.8)
    
    trace_low = go.Scatter(
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
    
    data = [trace_high,trace_low, trades_markers]
        
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





roc_auc_score(test['label'], test[0])





search_space = {
    'product': ['JPY_'+col for col in jpy.columns if len(col.split('Y')) > 2],
    'start_year': range(2015, 2019),
    'lookahead': [30],
}
results = []
for params in grid_gen(search_space):
    fe_params = copy(params)
    date_range = (datetime.datetime(fe_params.pop('start_year'),1,1), datetime.datetime(2020, 1, 1))
    product_name = fe_params.pop('product')
    pdata = ProductData(Product.from_string(product_name), dates=date_range)
    curve_fe = CurveFeatEng(**fe_params)
    df = curve_fe.get_features_labels(pdata)
    params['corr'] = np.corrcoef(df[df.columns[0]], df['label'])[0][1]
    results.append(params)
results_df = pd.DataFrame(results)


curve_fe = CurveFeatEng(5, 1)

curve_df = curve_fe.get_features_labels(pdata)

date_range = (datetime.datetime(2018,1,1), datetime.datetime(2019,1,1))
plot_df(filter_df(curve_df,dates=date_range))
plt.plot(curve_df[curve_df.columns[0]])
plt.plot(curve_df['label'])
plt.plot(pdata.product_series())







search_space = {
    'product': ['JPY_'+col for col in jpy.columns if len(col.split('Y')) > 2],
    'start_year': range(2015, 2018),
    'lb': range(15, 91, 15),
    'lookahead': [1, 5, 10],
}
results = []
for params in grid_gen(search_space):
    fe_params = copy(params)
    date_range = (datetime.datetime(fe_params.pop('start_year'),1,1), datetime.datetime(2020, 1, 1))
    product_name = fe_params.pop('product')
    pdata = ProductData(Product.from_string(product_name), dates=date_range)
    bol_fe = BollingerFeatEng(**fe_params)
    df = bol_fe.get_features_labels(pdata)
    params['corr'] = np.corrcoef(df['%b'], df['label'])[0][1]
    results.append(params)
results_df = pd.DataFrame(results)
    

    
    


bol_fe = BollingerFeatEng(15, 5)

bol_df = bol_fe.get_features_labels(pdata)

date_range = (datetime.datetime(2018,1,1), datetime.datetime(2019,1,1))
plot_df(filter_df(bol_df,dates=date_range))


series = pdata.product_series()
nearest = pdata.closest_fwd(-1)

near = pd.concat([pdata.closest_fwd(-i) for i in range(4)], axis=1)

fe = BollingerFeatEng(30,30, 5)

feat_lab = fe.get_features_labels(pdata)



time_fe = TimeSeriesFeatEng(5)

time_df = time_fe.get_features_labels(pdata)

date_range = (datetime.datetime(2015,1,1), datetime.datetime(2019,1,1))
plot_df(filter_df(time_df,dates=date_range))


series = pdata.product_series()
nearest = pdata.closest_fwd(-1)

near = pd.concat([pdata.closest_fwd(-i) for i in range(4)], axis=1)

fe = BollingerFeatEng(30,30, 5)

feat_lab = fe.get_features_labels(pdata)



df.loc[[datetime.datetime(2019,1,4)],:]
dates = [datetime.datetime(2019,1,4)]

df[df.index.map(lambda x: x in dates)]