#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:25:49 2020

@author: m

TODO:
    -spreads
    -plot the ma band, curr date, historic curr dates
    -highlight current term for forward curve
    -last refresh
"""
import os
os.chdir(os.path.abspath(os.path.join(__file__, os.pardir)))
import time
import datetime
import threading
import dateutil
import itertools

import numpy as np
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output

import xccy.data
from xccy.data import ProductData, Product
from xccy.modelling import MIN_DATA_DATE
from xccy.bounds import BollingerBand
from xccy.data import PAY, RECEIVE


SIDES = [PAY, RECEIVE]
CCYS = ['AUD', 'JPY', 'EUR', 'NZD', 'GBP']
ALL_TERMS = [f'{term}Y{fwd}Y' for term, fwd in itertools.product(range(1,6),range(1,6))]
DEFAULT_TERMS = ['1Y1Y', '2Y1Y', '3Y1Y', '1Y2Y', '2Y2Y', '5Y5Y']
STD_LB = [14, 30, 60, 90, 120, 150]
PERCENTILES = [50, 75, 90, 95, 100]
SEASON_WINDOW = 7
DEFAULT_MA_PERIOD = 60
DEFAULT_SEASON_LF = 30
SORT_BY = 'ma_score'

# data
DATA_DIR = 'data'
REFRSH_INTERVAL = 2 * 60 * 60
def refesh_data():
    try:
        xccy.data.refresh_data(DATA_DIR)
    except Exception as e:
        print('WARNING: Data not refreshed \n'
              '{}: {}'.format(type(e).__name__, e))
    xccy.data.initialize_data(DATA_DIR)
refesh_data()

def refresh_thread():
    while True:
        refesh_data()
        time.sleep(REFRSH_INTERVAL)

thread = threading.Thread(target=refresh_thread, args=())
thread.daemon = True   # Daemonize thread
thread.start() 
        
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/solar.csv')


app.layout = html.Div(children=[
        
    # html.H1(children='Hello Dash'),
    html.Div(children=[
        'Current: ',
        dcc.DatePickerSingle(
            id='date-picker',
            date=max(xccy.data.global_data.get_time_series()),
            display_format='DD/MM/YYYY'
            ),
    ]),
    html.Div(children=[
        'Starting: ',
        dcc.DatePickerSingle(
            id='min-date-picker',
            date=MIN_DATA_DATE,
            display_format='DD/MM/YYYY'
            ),
    ]),

    dcc.Dropdown(
        id='sides',
        options=[
            {'label': x, 'value': x} for x in SIDES
        ],
        multi=True,
        value=SIDES
    ),

    dcc.Dropdown(
        id='ccys',
        options=[
            {'label': x, 'value': x} for x in CCYS
        ],
        multi=True,
        value=CCYS
    ),
    
    dcc.Dropdown(
        id='terms',
        options=[
            {'label': x, 'value': x} for x in ALL_TERMS
        ],
        multi=True,
        value=DEFAULT_TERMS
    ),
            
    html.Div(children=[
        'MA period (days): ',
        dcc.Input(
            id='ma_period',
            placeholder='Moving average lookback period',
            type='number',
            value=DEFAULT_MA_PERIOD
        ),
    ]), 
    
    html.Div(children=[
        'Seasonality lookahead (days): ',
        dcc.Input(
            id='season_lf',
            placeholder='Seasonality lookahead period',
            type='number',
            value=DEFAULT_SEASON_LF
        ),
    ]),
    
    dash_table.DataTable(
        id='scores',
        row_selectable='single',
        # editable=True,
        style_data={'whiteSpace': 'normal', 'textAlign': 'center'},
        style_header={'textAlign': 'center'},
        css=[{
            'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }],
        sorting=True,
        sorting_type='multi',
        pagination_settings={
            'current_page': 0,
            'page_size': 10,
        },
        # filtering=True,
        data=[],
        columns=[],
        # style_data_conditional=state.table_formatting(),
    ),

    dash_table.DataTable(
        id='position',
        # row_selectable='single',
        style_data={'whiteSpace': 'normal', 'textAlign': 'center'},
        style_header={'textAlign': 'center'},
        css=[{
            'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }],
        #sorting=True,
        #sorting_type='multi',
        # filtering=True,
        data=[],
        columns=[],
        # style_data_conditional=state.table_formatting(),
    ),

    html.Div(id='graph-container'),
])

@app.callback(
    Output('scores', 'data'),
    [Input('date-picker', 'date'),
     Input('min-date-picker', 'date'),
     Input('sides', 'value'), 
     Input('ccys', 'value'), 
     Input('terms', 'value'),
     Input('ma_period', 'value'),
     Input('season_lf', 'value')])
def set_score_table(date, min_date, sides, ccys, terms, ma_period, season_lf):
    try:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d')
    except TypeError:
        date = None
    if date and ccys and terms and ma_period and season_lf:
        data = [get_scores(Product.from_string(f'{ccy}_{t}'), 
                           side, date, min_date, ma_period, season_lf) 
                for ccy, side, t in itertools.product(ccys, sides, terms)]
        data = sorted(data, key=lambda x: x[SORT_BY], reverse=True)
        return data
    return []

@app.callback(
    Output('position', 'data'),
    [Input('scores', 'derived_virtual_data'),
     Input('scores', 'selected_rows'),
     Input('date-picker', 'date')])
def set_position_table(rows, selected, date):
    try:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
    except TypeError:
        date = None
    if selected and selected[0] < len(rows) and date:
        row = rows[selected[0]]
        product = Product.from_string(row['ccy'] + '_' + row['product'])
        side = row['side']
        return get_position_data(product, side, date)
    return []

@app.callback(
    Output('graph-container', 'children'),
    [Input('scores', 'derived_virtual_data'),
     Input('scores', 'selected_rows'),
     Input('date-picker', 'date'),
     Input('min-date-picker', 'date')])
def update_graph(rows, selected, date, min_date):
    try:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d')
    except TypeError:
        date = None
    if selected and selected[0] < len(rows) and date:
        row = rows[selected[0]]
        product = Product.from_string(row['ccy'] + '_' + row['product'])
        return [get_series_graph(product, date, min_date), 
                get_fwd_graph(product, date)]

def sync_columns(data):
    def to_col(key, value):
        col = {'id': key, 'name': key}
        return col
    if data:
        return [to_col(k, v) for k, v in data[0].items()]
    return []

app.callback(Output('scores', 'columns'), [Input('scores', 'data')])(sync_columns)
app.callback(Output('position', 'columns'),[Input('position', 'data')])(sync_columns)

#funcs
def _get_current(product, date):
    return ProductData(product).series.loc[date]

def _get_ma_scores(product, side, date, lb):
    series = ProductData(product, 
                         max_date=date,
                         min_date=date - datetime.timedelta(days=90)).series
    band = BollingerBand(lb, lb, 1).get_bounds_df(series).loc[date]
    sd = (band['%b'] - 0.5) * 2
    # ma_score
    xp = np.array([0, 1.0, 1.50, 2])
    yp = np.array([0, 0.25, 0.75, 1])
    sign = 1 if side == RECEIVE else -1
    ma_score = np.interp(sign*sd, xp, yp)
    return {
        'ma_score': ma_score,
        'z': sd,
        str(lb)+'ma': band['ma'],
        'stddev': band['ma'] - band['lower']
    }


def _get_season_scores(product, side, date, min_date, lf_days):
    series = ProductData(product, 
                         max_date=date,
                         min_date=min_date).series
    series = series.rolling(SEASON_WINDOW, center=True).mean().dropna()

    def hist_diff_series(date, lf_days):
        dates = []
        values = []
        date = date - dateutil.relativedelta.relativedelta(years=1)
        lf_date = date + dateutil.relativedelta.relativedelta(days=lf_days)
        while min(series.index) < date and lf_date < max(series.index):
            ref = series.iloc[series.index.get_loc(date, method='nearest')]
            lf = series.iloc[series.index.get_loc(lf_date, method='nearest')]
            value = lf - ref
            if value:
                values.append(value)
                dates.append(date)
            date = date - dateutil.relativedelta.relativedelta(years=1)
            lf_date = date + dateutil.relativedelta.relativedelta(days=lf_days)
        return pd.Series(values, index=dates)
    
    diffs = hist_diff_series(date, lf_days)
    mean_diff = diffs.mean()
    median_diff = diffs.median()
    score = mean_diff if side == PAY else -mean_diff
    return {'season_score': score, 
            f'mean_{lf_days}day_change': mean_diff,
            f'median_{lf_days}day_change': median_diff}
    

def get_scores(product, side, date, min_date, ma_period, season_lf):
    output = {
        'side': side,
        'ccy': product.ccy.upper(),
        'product': product.to_string(ccy=False)}
    output['bp'] = _get_current(product, date)
    output.update(_get_ma_scores(product, side, date, ma_period))
    output.update(_get_season_scores(product, side, date, min_date, season_lf))
    for k, v in output.items():
        if isinstance(v, float):
            output[k] = round(v, 3)
    return output

def get_position_data(product, side, date):
    data = []
    for lb in STD_LB:
        row = {'lookback period (days)': lb}
        series = ProductData(product, 
                             max_date=date,
                             min_date=date - datetime.timedelta(days=lb)).series
        std = series.std()
        row['stdev'] = round(std, 3)
        diffs = (series.shift(1) - series).dropna()
        downs = np.array(diffs[diffs < 0])
        ups = np.array(diffs[diffs > 0])
        for array, sign in [(ups, '+'), (downs, '-')]:
            row['%'+sign] = round((len(array) / len(diffs)) * 100, 1)
            for p in PERCENTILES:
                label = {50: 'median', 100: 'max'}.get(p, f'{p}%') + sign
                row[label] = round(np.quantile(array, p/100), 3) if len(array) else 0
        data.append(row)
    return data

def get_series_graph(product, date, min_date):
    series = ProductData(
                 product, 
                 #max_date=date,
                 min_date=min_date,
             ).series
    data = [{'x': list(series.index),
             'y': list(series),
             'line': 'spline'}]
    return dcc.Graph(
        id='series-graph',
            figure={
                'data': data,
                'layout': {
                    'title': product.to_string()
                }
            }
        )


_fwd_graph_cache = {}

def get_fwd_graph(product, date):
    key = (product, product.term, date)
    if key in _fwd_graph_cache:
        x, y = _fwd_graph_cache[key]
    else: # make fwds
        pdata = ProductData(product, max_date=date, min_date=date)
        series = pdata.series
        fwds = {series.name: series.loc[date]}
        def _add_fwds(side, max_=8):
            for i in range(1, max_+1):
                series = pdata.closest_fwd(side*i)
                if series is not None:
                    fwds[series.name] = series.loc[date]
                else:
                    break
        _add_fwds(1)
        _add_fwds(-1)
        def sort_func(f):
            split = f[0].split('Y')
            if len(split) < 3:
                return 0
            else:
                return int(split[0])
        x, y = zip(*list(sorted(fwds.items(), key=sort_func)))
        _fwd_graph_cache[key] = (x, y)
    data = [{'x': list(x),
             'y': list(y),
             'line': 'spline'}]
    return dcc.Graph(
        id='fwd-graph',
            figure={
                'data': data,
                'layout': {
                    'title': f'Forward Curve ({product.to_string()})'
                }
            }
        )
    

if __name__ == '__main__':
    app.run_server(port=8080, debug=True)