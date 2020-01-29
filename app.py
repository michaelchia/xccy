#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:25:49 2020

@author: m

TODO:
    select ma period
    better chart
"""
import datetime
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


CCYS = ['AUD', 'JPY', 'EUR', 'NZD', 'GBP']
ALL_TERMS = [f'{term}Y{fwd}Y' for term, fwd in itertools.product(range(1,6),range(1,6))]
DEFAULT_TERMS = ['1Y1Y', '2Y1Y', '3Y1Y', '1Y2Y', '2Y2Y', '5Y5Y']
STD_LB = [14, 30, 60, 90, 120, 150]
PERCENTILES = [50, 75, 90, 95, 100]

xccy.data.initialize_data("data")

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
    
    dcc.RadioItems(
        id='side',
        options=[
            {'label': 'Pay', 'value': PAY},
            {'label': 'Receive', 'value': RECEIVE},
        ],
        value=None,
        labelStyle={'display': 'inline-block'}
    ),

    dcc.RadioItems(
        id='ccy',
        options=[
            {'label': x, 'value': x} for x in CCYS
        ],
        value=None,
        labelStyle={'display': 'inline-block'}
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
            value=60
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
        #sorting=True,
        #sorting_type='multi',
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
     Input('side', 'value'), 
     Input('ccy', 'value'), 
     Input('terms', 'value'),
     Input('ma_period', 'value')])
def set_score_table(date, min_date, side, ccy, terms, ma_period):
    try:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d')
    except TypeError:
        date = None
    if date and side and ccy and terms and ma_period:
        data = [get_scores(Product.from_string(f'{ccy}_{t}'), 
                           side, date, min_date, ma_period) for t in terms]
        return data
    return []

@app.callback(
    Output('position', 'data'),
    [Input('scores', 'derived_virtual_data'),
     Input('scores', 'selected_rows'),
     Input('date-picker', 'date'),
     Input('side', 'value')])
def set_position_table(rows, selected, date, side):
    try:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
    except TypeError:
        date = None
    if selected and selected[0] < len(rows) and date and side:
        row = rows[selected[0]]
        product = Product.from_string(row['product'])
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
        product = Product.from_string(row['product'])
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
    yp = np.array([0, 0.0, 0.25, 1])
    sign = 1 if side == RECEIVE else -1
    ma_score = np.interp(sign*sd, xp, yp)
    return {
        'ma_score': ma_score,
        'z': sd,
        str(lb)+'ma': band['ma'],
        'stddev': band['ma'] - band['lower']
    }
    
def _get_season_scores(product, side, date, min_date):
    series = ProductData(product, 
                         max_date=date,
                         min_date=min_date).series
    season_df = pd.DataFrame({"series": series, 
                             "month": series.index.map(lambda x: x.month),
                             "year": series.index.map(lambda x: x.year)}, index=series.index)
    m_season_df = season_df.groupby(["year", "month"]).mean().reset_index()
    m_season_df["avg_mth_change"] = m_season_df["series"].shift(-1) -  m_season_df["series"]
    m_season_df = m_season_df.groupby("month").mean()[["avg_mth_change"]].reset_index()
    def score(x):
        sign = 1 if side == PAY else -1
        x = sign*x
        max_ = max((sign*m_season_df["avg_mth_change"]).max(), 1e-5)
        return max(x / max_, 0)
    m_season_df["season_score"] = m_season_df["avg_mth_change"].map(score)
    m_season_df = m_season_df.set_index("month")
    return m_season_df.loc[date.month][["season_score", "avg_mth_change"]].to_dict()

def get_scores(product, side, date, min_date, ma_period):
    output = {'product': product.to_string()}
    output['bp'] = _get_current(product, date)
    output.update(_get_ma_scores(product, side, date, ma_period))
    output.update(_get_season_scores(product, side, date, min_date))
    for k, v in output.items():
        if isinstance(v, float):
            output[k] = round(v, 3)
    return output

def get_position_data(product, side, date):
    data = []
    sign = -1 if side == PAY else 1
    for lb in STD_LB:
        row = {'lookback period (days)': lb}
        series = ProductData(product, 
                             max_date=date,
                             min_date=date - datetime.timedelta(days=lb)).series
        std = series.std()
        row['stdev'] = round(std, 3)
        for p in PERCENTILES:
            losses = (series.shift(1) - series) * sign
            losses = np.array(losses[losses > 0])
            label = 'max' if p == 100 else f'{p}%'
            row[label] = round(np.quantile(losses, p/100), 3) * sign if len(losses) else 0
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