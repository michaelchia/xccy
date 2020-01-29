#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 00:41:24 2019

@author: m
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 02:59:08 2019

@author: m
"""
import math

import pandas as pd


def get_delta(series, lb):
    return series - series.shift(lb, freq=pd.DateOffset(days=1))

def get_ma(series, lb):
    return series.rolling(window='{}D'.format(lb), center=False, closed='both').mean()

def get_std(series, lb):
    return get_delta(series, 1).rolling(window='{}D'.format(lb), center=False, closed='both').std()


class MaBand:
    def __init__(self, ma_lb, sd_lb, x, lookahead=1):
        self.ma_lb = int(ma_lb)
        self.sd_lb = int(sd_lb)
        self.x = float(x)
        self.lookahead = lookahead
    
    def get_bounds_df(self, series):
        ma = self.get_ma(series)
        sd = self.get_sd(series)
        upper = ma + sd * self.x
        lower = ma - sd * self.x
        b = (series - lower) / (upper - lower)
        return pd.DataFrame({
            'ma': ma,
            'upper': upper,
            'lower': lower,
            '%b': b,
            'bandwidth': upper-lower
        })
    
    def get_bounds_at(self, series, date):
        bounds_df = self.get_bounds_df(series)
        row = bounds_df.loc[date]
        return dict(row)
    
    def get_ma(self, series):
        return series.rolling(window='{}D'.format(self.ma_lb), center=False, closed='both').mean()
    
    def set_params(self, ma_lb, sd_lb, x):
        self.ma_lb = int(ma_lb)
        self.sd_lb = int(sd_lb)
        self.x = float(x)
        return self
        
    def get_params(self):
        return [
            ('MA period', self.ma_lb),
            ('SD period', self.sd_lb),
            ('x SD', self.x),
            ('Lookahead', self.lookahead)
        ]

class MinMaxBand(MaBand):
    name = 'MinMax'
    def __init__(self, ma_lb, lb, perc, x, lookahead=5):
        self.ma_lb = int(ma_lb)
        self.lb = int(lb)
        self.perc = float(perc)
        self.x = float(x)
        self.lookahead = lookahead
        
    def get_bounds_df(self, series):
        ma = self.get_ma(series)
        upper = series.shift(1).rolling(window='{}D'.format(self.lb), center=False, closed='both').quantile(max(self.perc, 0.5))
        lower = series.shift(1).rolling(window='{}D'.format(self.lb), center=False, closed='both').quantile(min(1 - self.perc, 0.5))
        upper = (upper - ma) * self.x + ma
        lower = (lower - ma) * self.x + ma
        b = (series - lower) / (upper - lower)
        return pd.DataFrame({
            'ma': ma,
            'upper': upper,
            'lower': lower,
            '%b': b,
            'bandwidth': upper-lower
        })
    
    def set_params(self, ma_lb, lb, perc, x):
        self.ma_lb = int(ma_lb)
        self.lb = int(lb)
        self.perc = float(perc)
        self.x = float(x)
        return self
    
    def get_params(self):
        return [
            ('MA period', self.ma_lb),
            ('Lookback period', self.lb),
            ('Percentile', self.perc),
            ('Scale', self.x),
            ('Lookahead', self.lookahead)
        ]
    
class BollingerBand(MaBand):
    name = 'Bollinger Bands'
    def get_sd(self, series):
        return series.rolling(window='{}D'.format(self.sd_lb), center=False, closed='both').std()

class DeviationBand(MaBand):    
    name = 'Deviation Bands'
    def get_sd(self, series):
        ma = self.get_ma(series)
        sq_dev = (series - ma).map(lambda x: x**2).rolling(window='{}D'.format(self.sd_lb), center=False, closed='both').mean()
        return sq_dev.map(math.sqrt)
