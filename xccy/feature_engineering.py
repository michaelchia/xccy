#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 00:33:55 2019

@author: m
"""
from collections import defaultdict
import itertools

import pandas as pd
import datetime
import dateutil
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


from xccy.bounds import BollingerBand, DeviationBand, get_ma
from xccy.data import filter_df

LABEL_COL = 'label'

class _FeatEng:
    def __init__(self, feat_eng, lab_eng):
        self.feat_eng = feat_eng
        self.lab_eng = lab_eng

    def get_features_labels(self, product_data):
        features = self.feat_eng.get_features(product_data)
        labels = self.lab_eng.get_labels(product_data)
        return features.join(labels).dropna()


# le
class LabelEng:
    label_column = LABEL_COL
    
    def __init__(self, lookahead, window=5, cost=-2):
        self.lookahead = lookahead
        self.window = window
        self.cost = cost
        
    def _get_lookahead_series(self, product_data):
        series = product_data.product_series()
        labels = series.rolling(self.window, center=True).mean().shift(-self.lookahead) - series + self.cost
        labels.name = LABEL_COL
        return labels   
        
        
class RegLabel(LabelEng):
    def get_labels(self, product_data):
        return self._get_lookahead_series(product_data)
    

class BinaryLabel(LabelEng):
    def __init__(self, lookahead, window=5, min_change=0, cost=-2):
        self.lookahead = lookahead
        self.window = window
        self.cost = cost
        self.min_change = min_change
        
    def get_labels(self, product_data):
        labels = self._get_lookahead_series(product_data) 
        return labels.map(lambda x: int(x > self.min_change))


class MultiLabel(LabelEng):
    def __init__(self, lookahead, window=5, min_change=5, cost=-2):
        self.lookahead = lookahead
        self.window = window
        self.cost = cost
        self.min_change = min_change
        
    def get_labels(self, product_data):
        labels = self._get_lookahead_series(product_data) 
        def lab(x):
            if x > abs(self.min_change):
                return 1
            if x < -abs(self.min_change):
                return 0
            return 2
        return labels.map(lambda x: x > self.min_change)


# fe


        
class MaDiffFeatEng:
    def __init__(self, lb=30, diff=15):
        self.lb = lb
        self.diff = diff
        
    def get_features(self, product_data):
        series = product_data.product_series()
        short_ma = get_ma(series, self.lb)
        long_ma = get_ma(series, self.lb + self.diff)
        ma_diff = short_ma - long_ma
        ma_diff.name = 'ma_diff'
        return ma_diff.to_frame()


class BandFeatEng:
    def __init__(self, lb=30, type='bol'):
        self.lb = lb
        self.type = type
        
    def get_features(self, product_data):
        fe = BollingerFeatEng(self.lb) if self.type == 'bol' else DeviationFeatEng(self.lb)
        return fe.get_features(product_data)
        

class BollingerFeatEng:
    def __init__(self, lb):
        self.lb = lb
        
    def get_features(self, product_data):
        series = product_data.product_series()
        df = BollingerBand(self.lb, self.lb, 1).get_bounds_df(series)
        return df[['%b']]
    

class DeviationFeatEng:
    def __init__(self, lb):
        self.lb = lb
        
    def get_features(self, product_data):
        series = product_data.product_series()
        df = DeviationBand(self.lb, self.lb, 1).get_bounds_df(series)
        return df[['%b']]


class TimeFeatEng:
    def __init__(self, quarter=True, month=True, pos_in_quarter=True):
        self.quarter = quarter
        self.month = month
        self.pos_in_quarter = pos_in_quarter
        
    def get_features(self, product_data):
        def get_quarter_pos(date):
            q = int((date.month - 1) / 3)
            m_start = q * 3 + 1
            start = datetime.datetime(date.year, m_start, 1) \
                if m_start <= 12 else datetime.datetime(date.year + 1, m_start - 12, 1)
            end = start + dateutil.relativedelta.relativedelta(months=3)
            t_days = (end - start).days
            days = (date - start).days
            return days / t_days
        dates = list(product_data.product_series().index)
        feats = {}
        if self.quarter:
            feats['quarter'] = [int((d.month-1) / 3) + 1 for d in dates]
        if self.month:
            feats['month'] = [d.month for d in dates]
        if self.pos_in_quarter:
            feats['pos_in_q'] = [get_quarter_pos(d) for d in dates]
        df = pd.DataFrame(feats, index=dates)
        return df


class CurveFeatEng:
    def __init__(self, closest_n=1, lb=30):
        self.closest_n = closest_n
        self.lb = lb
        
    def get_features(self, product_data):
        series = product_data.product_series()
        ls = []
        for i in range(1, self.closest_n+1):
            fwd = product_data.closest_fwd(-i)
            if fwd is not None:
                s = DeviationBand(self.lb, self.lb, 1).get_bounds_df(fwd - series)['%b']
                s.name = 'dev_%b_{}-{}'.format(fwd.name, series.name)
                ls.append(s)
        if ls:
            return pd.concat(ls, axis=1)
        return (series / 0).to_frame()


class _CurveFeatEng:
    def __init__(self, closest_n=1):
        self.closest_n = closest_n
        
    def get_features(self, product_data):
        series = product_data.product_series()
        ls = []
        for i in range(1, self.closest_n+1):
            fwd = product_data.closest_fwd(-i)
            if fwd is not None:
                s = fwd - series
                s.name = '{}-{}'.format(fwd.name, series.name)
                ls.append(s)
        if ls:
            return pd.concat(ls, axis=1)
        return (series / 0).to_frame()

class FeatEng:
    PARAM_SPACE = [
        (MaDiffFeatEng, {'lb': [5, 15], 'diff': [15, 30, 45]}),
        (BandFeatEng, {'lb': [30, 60, 90], 'type': ['bol', 'dev']}),
        (CurveFeatEng, {'closest_n': [1], 'lb': [30, 60, 90]}),
        #(TimeFeatEng, {})
    ]
        
    def __init__(self):
        self.feat_engs = defaultdict(list)
        for fe_cls, param_space in self.PARAM_SPACE:
            label = fe_cls.__name__.lower().replace('feateng', '')
            for params in grid_gen(param_space):
                self.feat_engs[label].append((fe_cls(**params), params))
    
    def get_features(self, product_data):
        dfs = []
        for label, fe_params in self.feat_engs.items():
            for fe, params in fe_params:
                df = fe.get_features(product_data)
                params_str = '$'.join('{}={}'.format(k, v) for k, v in params.items())
                def rename(col):
                    ls = [label,col,params_str] if params_str else [label,col]
                    return '__'.join(ls)
                df = df.rename(rename, axis='columns')
                dfs.append(df)
        df = filter_df(df, product_data.dates)
        return pd.concat(dfs, axis=1)

# fe selectors
class BaseFeatSelector(BaseEstimator, TransformerMixin):
    @property
    def prefix(self):
        return self.__class__.__name__.lower().replace('featselector', '')
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X):
        params_set = {'{}={}'.format(k, v) for k, v in self.get_params().items()}
        cols = [col for col in X.columns 
                if not (col.split('__')[0] == self.prefix 
                        and not set(col.split('__')[-1].split('$')) == params_set)]
        return X[cols]
    
class MaDiffFeatSelector(BaseFeatSelector, MaDiffFeatEng):
    pass

class BandFeatSelector(BaseFeatSelector, BandFeatEng):
    pass

class CurveFeatSelector(BaseFeatSelector, CurveFeatEng):
    pass

class TimeFeatSelector(BaseFeatSelector, TimeFeatEng):
    def transform(self, X):
        params = [param for param, flag in self.get_params() if flag]
        cols = [col for col in X.columns 
                if not (col.split('__')[0] == self.prefix 
                        and not col.split('__')[-1] in params)]
        return X[cols]

class FeatSelector(Pipeline):
    name = 'fs'
    
    def __init__(self):
        steps = [
            MaDiffFeatSelector(),
            BandFeatSelector(),
            CurveFeatSelector(),
        ]
        steps = [(s.prefix, s) for s in steps]
        super().__init__(steps)
    
    @classmethod
    def param_space(cls):
        param_space = {}
        for fe_cls, params in FeatEng.PARAM_SPACE:
            label = fe_cls.__name__.lower().replace('feateng', '')
            for param, values in params.items():
                param_space['__'.join([cls.name, label, param])] = values
        return param_space
                
#
def grid_gen(search_space):
    return ({k:v for k, v in zip(search_space.keys(), params)} 
            for params in itertools.product(*search_space.values()))
    
def _fe_label(fe):
    return fe.__class__.__name__.lower().replace('feateng', '')