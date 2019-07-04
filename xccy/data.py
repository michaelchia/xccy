#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 01:07:30 2019

@author: m
"""
import math
import glob
import os

import datetime
import pandas as pd

PAY = 'pay'  # buy
RECEIVE = 'receive'  # sell

DPW = 7
DPM = 30
MPY = 12
DPY = DPM*MPY
DPX = {
    'Y': DPY,
    'M': DPM,
    'W': DPW,
    'D': 1
}


global_data = None
def initialize_data(data_path):
    global global_data
    df_dict = _read_files(data_path)
    global_data = GlobalXccyData(**df_dict)


def _read_files(data_path):
    FILE_EXTENSION = '.csv'
    if not data_path.endswith(os.sep):
        data_path += os.sep
    data_path += '*' + FILE_EXTENSION
    files = glob.glob(data_path)
    def namer(filepath): return os.path.basename(filepath.split(FILE_EXTENSION, 1)[0])
    def reader(filepath): return _preprocess_src_df(pd.read_csv(filepath))
    df_dict = {namer(f): reader(f) for f in files}
    return df_dict
 
def _preprocess_src_df(df):    
    df = df[df.columns[:14]].copy()
    df.loc[:,'Date'] = df['Date'].map(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y'))
    df = df.set_index('Date')
    def rename_column(col):
        token = col.split(' ', 1)[0]
        try:
            i = next(i for i, c in enumerate(token) if c.isdigit())
        except StopIteration:
            return '3M'
        label = token[i:].upper()
        if label[-1].isdigit():
            label += 'Y'
        return label
    df = df.apply(pd.to_numeric)
    df = df.dropna()
    df = df.rename(columns=rename_column)
    df = df.sort_index()
    return df


class GlobalXccyData:
    MAX_SPOT = 10 * DPY
    
    def __init__(self, **src_dfs):
        self._xccy_data_dict = {k: LocalXccyData(v) for k, v in src_dfs.items()}
        
    def get_time_series(self):
        # assumes all src_dfs have same indices
        return list(self._xccy_data_dict.values())[0].get_time_series()
    
    def get_series(self, product, dates=None):
        return self.local_data(product.ccy).get_series(product, dates)
    
    def get_bp(self, product, date):
        return self.local_data(product.ccy).get_bp(product, date)
    
    def local_data(self, ccy):
        return self._xccy_data_dict[ccy]


class LocalXccyData:
    def __init__(self, src_df, cache_derivations=True):
        self._src_df = src_df
        self.cache_derivations = cache_derivations
        
    def get_time_series(self):
        return list(self._src_df.index)
    
    def get_series(self, product, dates=None):
        src_df = self._filtered_src_df(dates)            
        name = product.to_string(ccy=False)
        if name in src_df:
            return src_df[name]
        if product.is_spot:
            raise KeyError('product %s not in data' % product.__repr__())
        if product.is_derivable:
            return self._simple_derivation(product, dates)
        return self._interpolate(product, dates)
    
    def get_bp(self, product, date):
        return self.get_series(product, [date]).iloc[0]
    
    def _simple_derivation(self, product, dates):
        if self.cache_derivations:
            orig_dates = dates
            dates = None
        product1, product2 = _component_spots(product)
        spot1  = self.get_series(product1, dates)
        spot2 = self.get_series(product2, dates)
        series = pd.Series(
            get_fwd(spot1, spot2, product1.term, product2.term),
            index=spot1.index)
        series = series.rename(product.to_string(ccy=False))
        if self.cache_derivations:
            self._src_df[product.to_string(ccy=False)] = series
            return self.get_series(product, orig_dates)
        return series
        
    def _interpolate(self, product, dates):
        product1, product2 = _nearest_fowards(product)
        fwd1 = self.get_series(product1, dates)
        fwd2 = self.get_series(product2, dates)
        assert product1.fwd <= product.fwd <= product2.fwd 
        assert product1.term == product.term == product2.term 
        dist = (product.fwd - product1.fwd) / (product2.fwd - product1.fwd)
        return fwd1 * (1 - dist) + fwd2 * dist
        
    def _filtered_src_df(self, dates):
        return filter_df(self._src_df, dates)


class Product:
    def __init__(self, fwd, term, ccy=None, period='Y'):
        days = DPX[period]
        self.fwd = fwd * days
        self.term = term * days
        self.ccy = ccy.upper()
    
    def to_string(self, ccy=True):
        ccy = self.ccy.upper() + '_' if ccy and self.ccy else ''
        def days_to_string(days):
            if not days:
                return ''
            if not days % DPY:
                label = '%dY' % (days/DPY)
            elif not days % DPM:
                label = '%dM' % (days/DPM)
            elif days < DPM and not days % DPW:
                label = '%dW' % (days/DPW)
            else:
                label = '%dD' % days
            return label
        return ccy + days_to_string(self.fwd) + days_to_string(self.term)
    
    def set_fwd(self, fwd, period='D'):
        days = DPX[period]
        self.fwd = fwd * days
        return self
    
    def set_term(self, term, period='D'):
        days = DPX[period]
        self.term = term * days
        return self
    
    def age_product(self, delta, period='D'):
        days = DPX[period]
        self.fwd -= delta * days
        return self
    
    def copy(self):
        return Product(self.fwd, self.term,
                       ccy=self.ccy, period='D')
        
    @property
    def total_period(self):
        return self.fwd + self.term
        
    @property
    def is_derivable(self):
        return self.fwd > 0 and (not self.fwd % DPY) and (not self.term % DPY)
    
    @property
    def is_spot(self):
        return self.fwd == 0
    
    @classmethod
    def from_string(cls, str_):
        tokens = []
        t = ''
        for c in str_.upper():
            if not t or c.isdigit() == t.isdigit():
                t += c
            else:
                tokens.append(t)
                t = c
        tokens.append(t)
        if not tokens[0].isdigit():
            ccy = tokens.pop(0).replace('_','')
        else:
            ccy = None
        assert len(tokens) == 2 or len(tokens) == 4
        if len(tokens) == 2:
            fwd = 0
            term, t_period = tokens
            term = int(term) * DPX[t_period]
        if len(tokens) == 4:
            fwd, f_period, term, t_period = tokens
            fwd = int(fwd) * DPX[f_period]
            term = int(term) * DPX[t_period]
        return cls(fwd, term, ccy=ccy, period='D')
                
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.to_string(ccy=True))
    
    def __str__(self):
        return self.to_string()
    

class ProductData:
    def __init__(self, product, dates=None):
        if global_data is None:
            raise Exception('initialise data first')
        self.product = product
        self.dates = dates
        
    def product_series(self, dates=None):
        return global_data.get_series(self.product, dates)
    
    def closest_fwd(self, n=-1, dates=None):
        if self.product.term < DPY:
            raise NotImplementedError()
        return self._rec_closest_fwd(self.product, int(n), dates)
    
    def _rec_closest_fwd(self, product, n, dates):
        if n == 0:
            return global_data.get_series(product, dates)
        inc = 1 if n > 0 else -1
        rnd = math.floor if n < 0 else math.ceil
        closest_fwd = (rnd(product.fwd / DPY) + inc) * DPY
        while 0 <= closest_fwd <= global_data.MAX_SPOT:
            _product = self.product.copy().set_fwd(closest_fwd)
            try:
                series = global_data.get_series(_product, self.dates)
                if n - inc == 0:
                    return series
                return self._rec_closest_fwd(_product, n - inc)
            except KeyError:
                closest_fwd += inc
        return None        


class Trade:
    def __init__(self, product, side, enter_date):
        if global_data is None:
            raise Exception('initialise data first')
        self.product = product
        self.side = side
        self.enter_date = enter_date
        
    @property
    def start_bp(self):
        return global_data.get_bp(self.product, self.enter_date)
    
    def pl_at(self, date):
        sign = 1 if self.side == PAY else -1
        return (self.bp_at(date) - self.start_bp) * sign
    
    def bp_at(self, date):
        product = self.product_at(date)
        return global_data.get_bp(product, date)
        
    def product_at(self, date):
        assert date >= self.enter_date
        delta = (date - self.enter_date).days
        return self.product.copy().age_product(delta, period='D')
        

def _component_spots(product):
    product1 = product.copy().set_term(product.fwd).set_fwd(0)
    product2 = product.copy().set_term(product.fwd + product.term).set_fwd(0)
    return product1, product2


def _nearest_fowards(product):
    if product.term < DPY:
        raise NotImplementedError()
    factor = DPY
    fwd_y = product.fwd / factor
    l, u = math.floor(fwd_y), math.ceil(fwd_y)
    product1 = product.copy().set_fwd(l * factor)
    product2 = product.copy().set_fwd(u * factor)
    return product1, product2

def filter_df(df, dates):
    if isinstance(dates, tuple) and len(dates) == 2:
        a, b = dates
        return df[df.index.map(lambda x: a <= x <=b)]
    if dates:
        return df[df.index.map(lambda x: x in dates)]
    return df
    

# TODO: vectorize these
def get_spot(spot, fwd, m1, m2):
    zipped = zip(spot, fwd)
    return [round((f * (m2 - m1) + s * m1) / m2, 2) for s, f in zipped]

def get_fwd(spot1, spot2, m1, m2):
    zipped = zip(spot1, spot2)
    return [round((s2 * m2 - s1 * m1) / (m2 - m1), 2) for s1, s2 in zipped]